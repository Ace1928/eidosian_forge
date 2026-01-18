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
class FixedDialogTeacher(Teacher):
    """
    A teacher agent for all teachers involved in tasks with fixed data.

    This class provides the following functionality for its subclasses:

    - Resets a teacher
    - Provides an observe method
    - Computes and retrieves the next episode index for a teacher
    - Provides a threadpool option for loading data (especially useful for
      large data, e.g. images)

    In order to take advantage of the first few features, all a subclass has to
    implement is three functions: ``num_episodes``, ``num_examples``, and
    ``get`` (which returns a specific example from a specific episode).

    To utilize the DataLoader for threadpool loading, a teacher should
    implement the ``submit_load_request`` function to send a load request
    to the DataLoader by calling ``self.data_loader.request_load`` with the
    appropriate arguments (``receive_fn, load_fn, args``). The DataLoader then
    returns the data to the teacher's ``data_queue``, which the teacher can
    poll in its ``act`` method.

    The following is an example of the DataLoader usage in the VQA-V1 teacher.

    1. In the teacher's ``init`` function, the teacher calls its
       ``submit_load_request`` function to preload an image.
    2. The ``submit_load_request`` function gets the next ``episode_idx``,
       and computes the image path for the load request.
    3. At the end of ``submit_load_request``, the teacher calls
       ``self.data_loader.request_load`` with three args:

        - ``self.receive_data`` - the function that the DataLoader calls to
          return the the loaded object
        - ``self.image_loader.load`` - the function used to load the image
          from the image path
        - ``[img_path]`` - a list of arguments for the load function, which
          in this case is the path of the image.

    4. In the teacher's ``act`` function, the teacher loads the data from
       its data queue.
    5. At the end of the ``act`` function, the teacher calls
       ``submit_load_request`` to preload an image for the next example.

    To see this in action, take a look at this teacher in ``tasks.vqa_v1.agents``.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not hasattr(self, 'datatype'):
            self.datatype = opt['datatype']
        if not hasattr(self, 'random'):
            self.random = self.datatype == 'train'
        if not hasattr(self, 'training'):
            self.training = DatatypeHelper.is_training(self.datatype)
        if not hasattr(self, 'cycle'):
            self.cycle = DatatypeHelper.should_cycle(self.datatype)
        if not hasattr(self, 'datafile'):
            self.datafile = opt.get('datafile')
        self.data_queue = queue.Queue()
        if shared:
            self.index = shared['index']
            if 'data_loader' in shared:
                self.data_loader = shared['data_loader']
            if 'threadindex' in shared:
                self.threadindex = shared['threadindex']
            if 'examples' in shared:
                self.examples = shared['examples']
        else:
            self.index = AttrDict(value=-1)
        if not hasattr(self, 'data_loader'):
            self.data_loader = DataLoader(opt)
            self.data_loader.start()
        self.bsz = opt.get('batchsize', 1)

    def _lock(self):
        if hasattr(self.index, 'get_lock'):
            return self.index.get_lock()
        else:
            return no_lock()

    def reset(self):
        """
        Reset the dialog to the start of the epoch, and reset all metrics.
        """
        super().reset()
        self.metrics.clear()
        self.lastY = None
        self.last_act = None
        self._episode_done = True
        self.epochDone = False
        self.data_queue = queue.Queue()
        self.episode_idx = -1
        with self._lock():
            self.index.value = -1

    def submit_load_request(self):
        """
        Submit a load request.

        An agent should implement this method to submit requests to the data
        loader. At the end of this method, the agent should call
        ``self.data_loader.request_load()`` with the appropriate args.

        By default, this method does nothing.
        """
        pass

    def receive_data(self, future):
        """
        Receive data from the data loader.

        :param future: result from the load request.
        """
        data = future.result()
        self.data_queue.put(data)

    def share(self):
        """
        Share the data and dataloader.
        """
        shared = super().share()
        if hasattr(self, 'examples'):
            shared['examples'] = self.examples
        if hasattr(self, 'data_loader'):
            shared['data_loader'] = self.data_loader
        shared['index'] = self.index
        return shared

    def next_episode_idx(self, num_eps=None, loop=None):
        """
        Return the next episode index.

        :param num_eps:
            default None uses ``num_episodes`` value.
        :param loop:
            default None loops during training but not evaluation.
        """
        if num_eps is None:
            num_eps = self.num_episodes()
        if loop is None:
            loop = self.training
        if self.random:
            new_idx = random.randrange(num_eps)
        else:
            with self._lock():
                self.index.value += 1
                if loop:
                    self.index.value %= num_eps
                new_idx = self.index.value
        return new_idx

    def next_example(self):
        """
        Return the next example.

        If there are multiple examples in the same episode, returns the next one in that
        episode. If that episode is over, gets a new episode index and returns the first
        example of that episode.
        """
        if self._episode_done:
            self.episode_idx = self.next_episode_idx()
            self.entry_idx = 0
        else:
            self.entry_idx += 1
        if self.episode_idx >= self.num_episodes():
            return ({'episode_done': True}, True)
        ex = self.get(self.episode_idx, self.entry_idx)
        self._episode_done = ex.get('episode_done', False)
        if not self.cycle and self._episode_done and (self.episode_idx + self.opt.get('batchsize', 1) >= self.num_episodes()):
            epoch_done = True
        else:
            epoch_done = False
        return (ex, epoch_done)

    def next_batch(self):
        """
        Return the next batch of examples.
        """
        with self._lock():
            self.index.value += 1
            if self.training:
                self.index.value %= len(self.batches)
            batch_idx = self.index.value
            if batch_idx + 1 >= len(self.batches):
                if self.random:
                    random.shuffle(self.batches)
                self.epochDone = True
            else:
                self.epochDone = False
        if batch_idx >= len(self.batches):
            return [{'episode_done': True, 'id': self.getID()}] * self.bsz
        return self.batches[batch_idx]

    def num_episodes(self) -> int:
        """
        Get the number of episodes in this dataset.
        """
        raise RuntimeError('"num_episodes" must be overriden by children.')

    def num_examples(self) -> int:
        """
        Get the total number of examples in this dataset.
        """
        raise RuntimeError('"num_examples" must be overriden by children.')

    def get(self, episode_idx, entry_idx=0):
        """
        Get the specified episode and the specified entry in that episode.

        Children must override this method in order to inherit the
        `next_example` method.

        :param episode_idx:
            which episode to return examples from
        :param entry_idx:
            which example to return from the episode.  Many datasets have only
            single-entry episodes, so this defaults to zero.
        """
        raise RuntimeError('"Get" method must be overriden by children.')

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        if hasattr(self, 'lastY') and self.lastY is not None:
            self.metrics.evaluate_response(observation, self.lastY)
            self.custom_evaluation(self.last_act, self.lastY, observation)
            self.lastY = None
        return observation

    def custom_evaluation(self, teacher_action: Message, labels: Optional[Tuple[str]], model_response: Message) -> None:
        """
        A method designated for hooking custom evaluations into teachers.

        Generally, a user will want to use `self.metrics.add` to record any
        specialized metrics that only make sense for this one dataset.

        :param teacher_action:
            The message last sent from this teacher.
        :param labels:
            The previous correct labels, if there were any.
        :param model_response:
            The raw response from the model. Generally you want to rely on the
            text field, but others may be necessary in specific situations.
        """
        pass

    def act(self):
        """
        Send new dialog message.
        """
        if not hasattr(self, 'epochDone'):
            self.reset()
        action, self.epochDone = self.next_example()
        action = Message(action)
        action.force_set('id', self.getID())
        self.last_act = action
        self.lastY = action.get('labels', action.get('eval_labels', None))
        if not DatatypeHelper.is_training(self.datatype) and 'labels' in action:
            action = action.copy()
            labels = action.pop('labels')
            if not self.opt.get('hide_labels', False):
                action['eval_labels'] = labels
        return action