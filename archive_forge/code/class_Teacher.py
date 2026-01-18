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
class Teacher(Agent):
    """
    Basic Teacher agent that keeps track of how many times it's received messages.

    Teachers provide the ``report()`` method to get back metrics.
    """

    def __init__(self, opt: Opt, shared=None):
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        if not hasattr(self, 'id'):
            self.id = opt.get('task', 'teacher')
        if not hasattr(self, 'metrics'):
            self.metrics = TeacherMetrics(metrics_list=opt.get('metrics', 'default'), shared=shared['metrics'] if shared is not None else None)
        self.epochDone = False

    def act(self):
        """
        Act upon the previous observation.
        """
        if self.observation is not None and 'text' in self.observation:
            t = {'text': 'Hello agent!'}
        return t

    def epoch_done(self):
        """
        Return whether the epoch is done.
        """
        return self.epochDone

    def num_examples(self):
        """
        Return the number of examples (e.g. individual utterances) in the dataset.

        Default implementation returns `None`, indicating an unknown number.
        """
        return None

    def num_episodes(self):
        """
        Return the number of episodes (e.g. conversations) in the dataset.

        Default implementation returns `None`, indicating an unknown number.
        """
        return None

    def report(self):
        """
        Return metrics showing total examples and accuracy if available.
        """
        return self.metrics.report()

    def reset(self):
        """
        Reset the teacher.
        """
        super().reset()
        self.reset_metrics()
        self.epochDone = False

    def reset_metrics(self):
        """
        Reset metrics.
        """
        self.metrics.clear()

    def share(self):
        """
        In addition to default Agent shared parameters, share metrics.
        """
        shared = super().share()
        shared['metrics'] = self.metrics.share()
        return shared