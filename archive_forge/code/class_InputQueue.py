from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types
import numpy as np
import scipy.sparse
from smart_open import open
from gensim import __version__ as gensim_version
class InputQueue(multiprocessing.Process):
    """Populate a queue of input chunks from a streamed corpus.

    Useful for reading and chunking corpora in the background, in a separate process,
    so that workers that use the queue are not starved for input chunks.

    """

    def __init__(self, q, corpus, chunksize, maxsize, as_numpy):
        """
        Parameters
        ----------
        q : multiprocessing.Queue
            Enqueue chunks into this queue.
        corpus : iterable of iterable of (int, numeric)
            Corpus to read and split into "chunksize"-ed groups
        chunksize : int
            Split `corpus` into chunks of this size.
        as_numpy : bool, optional
            Enqueue chunks as `numpy.ndarray` instead of lists.

        """
        super(InputQueue, self).__init__()
        self.q = q
        self.maxsize = maxsize
        self.corpus = corpus
        self.chunksize = chunksize
        self.as_numpy = as_numpy

    def run(self):
        it = iter(self.corpus)
        while True:
            chunk = itertools.islice(it, self.chunksize)
            if self.as_numpy:
                wrapped_chunk = [[np.asarray(doc) for doc in chunk]]
            else:
                wrapped_chunk = [list(chunk)]
            if not wrapped_chunk[0]:
                self.q.put(None, block=True)
                break
            try:
                qsize = self.q.qsize()
            except NotImplementedError:
                qsize = '?'
            logger.debug('prepared another chunk of %i documents (qsize=%s)', len(wrapped_chunk[0]), qsize)
            self.q.put(wrapped_chunk.pop(), block=True)