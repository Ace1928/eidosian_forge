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
class SlicedCorpus(SaveLoad):
    """Wrap `corpus` and return a slice of it."""

    def __init__(self, corpus, slice_):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, numeric)
            Input corpus.
        slice_ : slice or iterable
            Slice for `corpus`.

        Notes
        -----
        Negative slicing can only be used if the corpus is indexable, otherwise, the corpus will be iterated over.
        Slice can also be a np.ndarray to support fancy indexing.

        Calculating the size of a SlicedCorpus is expensive when using a slice as the corpus has
        to be iterated over once. Using a list or np.ndarray does not have this drawback, but consumes more memory.

        """
        self.corpus = corpus
        self.slice_ = slice_
        self.length = None

    def __iter__(self):
        if hasattr(self.corpus, 'index') and len(self.corpus.index) > 0:
            return (self.corpus.docbyoffset(i) for i in self.corpus.index[self.slice_])
        return itertools.islice(self.corpus, self.slice_.start, self.slice_.stop, self.slice_.step)

    def __len__(self):
        if self.length is None:
            if isinstance(self.slice_, (list, np.ndarray)):
                self.length = len(self.slice_)
            elif isinstance(self.slice_, slice):
                start, end, step = self.slice_.indices(len(self.corpus.index))
                diff = end - start
                self.length = diff // step + (diff % step > 0)
            else:
                self.length = sum((1 for x in self))
        return self.length