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
class RepeatCorpus(SaveLoad):
    """Wrap a `corpus` as another corpus of length `reps`. This is achieved by repeating documents from `corpus`
    over and over again, until the requested length `len(result) == reps` is reached.
    Repetition is done on-the-fly=efficiently, via `itertools`.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import RepeatCorpus
        >>>
        >>> corpus = [[(1, 2)], []]  # 2 documents
        >>> list(RepeatCorpus(corpus, 5))  # repeat 2.5 times to get 5 documents
        [[(1, 2)], [], [(1, 2)], [], [(1, 2)]]

    """

    def __init__(self, corpus, reps):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, numeric)
            Input corpus.
        reps : int
            Number of repeats for documents from corpus.

        """
        self.corpus = corpus
        self.reps = reps

    def __iter__(self):
        return itertools.islice(itertools.cycle(self.corpus), self.reps)