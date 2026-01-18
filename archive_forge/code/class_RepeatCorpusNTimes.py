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
class RepeatCorpusNTimes(SaveLoad):
    """Wrap a `corpus` and repeat it `n` times.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import RepeatCorpusNTimes
        >>>
        >>> corpus = [[(1, 0.5)], []]
        >>> list(RepeatCorpusNTimes(corpus, 3))  # repeat 3 times
        [[(1, 0.5)], [], [(1, 0.5)], [], [(1, 0.5)], []]

    """

    def __init__(self, corpus, n):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, numeric)
            Input corpus.
        n : int
            Number of repeats for corpus.

        """
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            for document in self.corpus:
                yield document