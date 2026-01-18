from the disk or network on-the-fly, without loading your entire corpus into RAM.
from __future__ import division  # py3 "true division"
import logging
import sys
import os
import heapq
from timeit import default_timer
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from types import GeneratorType
import threading
import itertools
import copy
from queue import Queue, Empty
from numpy import float32 as REAL
import numpy as np
from gensim.utils import keep_vocab_item, call_on_class_only, deprecated
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
from gensim import utils, matutils
from gensim.models.keyedvectors import Vocab  # noqa
from smart_open.compression import get_supported_extensions
def _get_next_alpha(self, epoch_progress, cur_epoch):
    """Get the correct learning rate for the next iteration.

        Parameters
        ----------
        epoch_progress : float
            Ratio of finished work in the current epoch.
        cur_epoch : int
            Number of current iteration.

        Returns
        -------
        float
            The learning rate to be used in the next training epoch.

        """
    start_alpha = self.alpha
    end_alpha = self.min_alpha
    progress = (cur_epoch + epoch_progress) / self.epochs
    next_alpha = start_alpha - (start_alpha - end_alpha) * progress
    next_alpha = max(end_alpha, next_alpha)
    self.min_alpha_yet_reached = next_alpha
    return next_alpha