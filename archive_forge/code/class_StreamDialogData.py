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
class StreamDialogData(DialogData):
    """
    Provides a data structure for streaming textual dialog data.

    This can be used whenever the dialog data follows the format described in
    DialogData but cannot fit entirely into memory.

    Additional keyword-argument cycle defines if the stream should restart from
    the beginning after an epoch is finished (defaults to True).

    :param opt:
        options to initialize the class
    :param data_loader:
        an iterable with each call returning a tuple in the form
        ``((x, y, r, c, i), new_episode?)`` where the ``x`` and ``new_episode``
        fields are mandatory and other fields may be omitted or ``None``.
    :param cands:
        can be set to provide a list of candidate labels for every example in
        this dataset, which the agent can choose from (the correct answer
        should be in this set).
    :param random:
        tells the data class whether or not to visit episodes sequentially or
        randomly when returning examples to the caller.
    :param cycle:
        (default True) whether to restart at beginning when end of stream
        reached without reset being called.
    """
    _FIRST_PASS = None
    _END_OF_EPOCH = -1

    def __init__(self, opt, data_loader=None, cands=None, shared=None, **kwargs):
        super().__init__(opt, data_loader, cands, shared, **kwargs)
        self.cycle = kwargs['cycle'] if 'cycle' in kwargs else True
        if shared:
            self.reset_data = shared['reset']
            self.datafile = shared['datafile']
            self.data_loader = shared['data_loader']
            if 'lock' in shared:
                self.lock = shared['lock']
        else:
            self.data_loader = data_loader
            self.datafile = opt['datafile']
            self.reset_data = None
            self.is_reset = True
        self.entry_idx = 0
        self.cur_episode = self._FIRST_PASS
        self.num_eps = None
        self.num_exs = None
        self.rank = get_rank()
        self.num_workers = num_workers()
        self.is_distributed_and_is_eval = self.num_workers > 1 and any((x in opt['datatype'] for x in ('valid', 'test', 'train:evalmode')))

    def share(self):
        """
        Share the stream.
        """
        shared = super().share()
        shared['reset'] = self.reset
        shared['datafile'] = self.datafile
        shared['data_loader'] = self.data_loader
        if hasattr(self, 'lock'):
            shared['lock'] = self.lock
        return shared

    def _load(self, data_loader, datafile):
        """
        Load data generator into data field.
        """
        self.data = self._data_generator(data_loader, datafile)

    def _data_generator(self, data_loader, datafile):
        """
        Generate data using the iterator over tuples constructed by data_loader.
        """
        self.is_reset = False
        idx = 0
        while True:
            for episode in self._read_episode(data_loader(datafile)):
                if not self.is_distributed_and_is_eval or idx % self.num_workers == self.rank:
                    yield episode
                idx += 1
            while not self.cycle:
                yield self._END_OF_EPOCH

    def load_length(self):
        """
        Calculate the length of the dataset and caches it in a file.

        Note that this can take some time for large datasets. Episode and entry indexes
        cannot be specified during streaming.
        """
        datafiles = self.datafile if type(self.datafile) is tuple else [self.datafile]
        length_file = datafiles[0] + '.lengths'
        if not os.path.isfile(length_file):
            num_eps = 0
            num_exs = 0
            for episode in self._read_episode(self.data_loader(self.datafile)):
                num_eps += 1
                num_exs += len(episode)
            with open(length_file, 'w', encoding='utf-8') as f:
                f.write('{}\n{}'.format(num_eps, num_exs))
        else:
            with open(length_file, 'r', encoding='utf-8') as f:
                num_eps, num_exs = f.readlines()
        return (int(num_eps), int(num_exs))

    def num_examples(self):
        """
        Return the number of examples in the data.
        """
        if not self.num_exs:
            self.num_eps, self.num_exs = self.load_length()
        return self.num_exs

    def num_episodes(self):
        """
        Return the number of episodes in the data.
        """
        if not self.num_eps:
            self.num_eps, self.num_exs = self.load_length()
        return self.num_eps

    def _lock(self):
        if hasattr(self, 'lock'):
            return self.lock
        else:
            return no_lock()

    def get(self):
        """
        Get the next entry from the stream.

        When episode is done returns first entry of next episode.
        """
        if self.cur_episode is self._FIRST_PASS:
            self.cur_episode = next(self.data)
        if self.cur_episode == self._END_OF_EPOCH:
            return ({'episode_done': True}, True)
        entry = self.cur_episode[self.entry_idx]
        table = self.build_table(entry)
        episode_done = self.entry_idx == len(self.cur_episode) - 1
        table['episode_done'] = episode_done
        if episode_done:
            self.cur_episode = next(self.data)
            self.entry_idx = 0
        else:
            self.entry_idx += 1
        return (table, self.cur_episode == self._END_OF_EPOCH)

    def reset(self):
        """
        Reset the datastream to its beginning.
        """
        if self.reset_data is not None:
            self.data = self.reset_data()
        elif not self.is_reset:
            self._load(self.data_loader, self.datafile)
            self.is_reset = True
        self.entry_idx = 0
        self.cur_episode = self._FIRST_PASS
        return self.data