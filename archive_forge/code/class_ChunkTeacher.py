import copy
from typing import List, Tuple, Optional, TypeVar
from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.image_featurizers import ImageLoader
from parlai.core.loader import load_teacher_module
from parlai.core.loader import register_teacher  # noqa: F401
from parlai.core.message import Message
from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
from parlai.core.opt import Opt
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once
from parlai.utils.distributed import get_rank, num_workers, is_distributed
import parlai.utils.logging as logging
from abc import ABC, abstractmethod
import concurrent.futures
from threading import Thread
import queue
import random
import time
import os
import torch
import json
import argparse
class ChunkTeacher(FixedDialogTeacher, ABC):
    """
    Useful for loading large amounts of data.

    Data is separated into chunks and loaded one chunk at a time. Loads the data off of
    the main thread.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.buffersize = self.get_buffersize()
        if 'stream' not in opt['datatype']:
            raise ValueError('Chunk teacher should be used with streaming. ')
        self.set_datasettings(opt)
        self.dws = int(self.opt.get('distributed_world_size', 1))
        self.rank = int(self.opt.get('rank', 0))
        if shared is None and self.is_train and (self.opt.get('distributed_world_size') is not None):
            self.fold_chunks = [c for c in self.fold_chunks if c % self.dws == self.rank]
        if shared is not None:
            self.is_root_teacher = False
            self.chunks = shared['chunks']
            self.samples = shared['samples']
            self.rng = shared['rng']
        else:
            self.is_root_teacher = True
            self.samples = queue.Queue(maxsize=self.buffersize)
            self.chunks = queue.Queue()
            if self.is_train:
                self.rng = random.Random()
            else:
                self.rng = random.Random(42)
            self._enqueue_chunks()
            self.tot_samples_loaded = 0
            self._enqueue_request()
        self._episode_done = True
        self.last_queue_output = None

    def _get_data_folder(self):
        if not self.opt.get('datafile'):
            raise RuntimeError('Must specify datafile or override this function to return the data folder.')
        return self.opt['datafile']

    @abstractmethod
    def get_num_samples(self, opt: Opt) -> Tuple[int, int]:
        """
        [Abstract] Return the number of samples.

        Returns a tuple of (num_examples, num_episodes) based on the data split.
        """
        pass

    @abstractmethod
    def get_fold_chunks(self, opt: Opt) -> List[int]:
        """
        [Abstract] Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        pass

    def get_buffersize(self):
        """
        Size of buffer.

        Override this in your child class to change the buffer size.
        """
        return 100000

    def set_datasettings(self, opt: Opt):
        self.folder = self._get_data_folder()
        self.num_exs, self.num_eps = self.get_num_samples(opt)
        self.fold_chunks = self.get_fold_chunks(opt)
        self.is_train = DatatypeHelper.is_training(opt['datatype'])

    def share(self):
        shared = super().share()
        shared['samples'] = self.samples
        shared['chunks'] = self.chunks
        shared['rng'] = self.rng
        return shared

    def _setup_data(self, datatype):
        """
        Passthrough.
        """
        pass

    def num_episodes(self):
        if self.is_train:
            return self.num_eps
        else:
            return self.num_eps // self.dws + int(self.num_eps % self.dws > self.rank)

    def num_examples(self):
        if self.is_train:
            return self.num_exs
        else:
            return self.num_exs // self.dws + int(self.num_exs % self.dws > self.rank)

    def _enqueue_request(self):
        """
        Queue a request for loading to the data loader.
        """
        self.data_loader.request_load(self.receive_data, self.get_chunk, ())

    def receive_data(self, future):
        """
        Loads data.

        Load data into self.samples until buffersize is reached.
        """
        data = future.result()
        if data is None:
            return
        while data:
            sample = data.pop(0)
            if self.is_train or self.tot_samples_loaded % self.dws == self.rank:
                self.samples.put(sample)
            self.tot_samples_loaded += 1
        self._enqueue_request()

    def _enqueue_chunks(self):
        """
        Shuffles and queues fold chunks for loading.
        """
        if self.is_train:
            self.rng.shuffle(self.fold_chunks)
        for c in self.fold_chunks:
            self.chunks.put(c)

    @abstractmethod
    def load_from_chunk(self, chunk_idx: int) -> List[ChunkOutput]:
        """
        [Abstract] Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        pass

    @abstractmethod
    def create_message(self, queue_output: ChunkOutput, entry_idx=0) -> 'Message':
        """
        [Abstract] Given the tuple output of the queue, return an act.

        May depend on entry index if queue output is a multi-turn episode.
        """
        pass

    def get_chunk(self):
        """
        Refill the buffer.
        """
        if self.chunks.empty():
            if self.is_train:
                self._enqueue_chunks()
            else:
                return None
        next_chunk = self.chunks.get()
        output = self.load_from_chunk(next_chunk)
        if self.is_train:
            random.Random().shuffle(output)
        return output

    def get(self, episode_idx, entry_idx=0):
        if self._episode_done:
            queue_output = self.samples.get()
            if queue_output is None:
                return None
            self.last_queue_output = queue_output
        msg = self.create_message(self.last_queue_output, entry_idx)
        self._episode_done = msg['episode_done']
        return msg

    def _drain(self, q):
        while not q.empty():
            try:
                q.get()
            except queue.Empty:
                return

    def reset(self):
        super().reset()
        if self.is_root_teacher:
            self._drain(self.samples)
            self._drain(self.chunks)
            self._enqueue_chunks()
            self.tot_samples_loaded = 0
            self._enqueue_request()