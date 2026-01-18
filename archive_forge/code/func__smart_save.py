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
def _smart_save(self, fname, separately=None, sep_limit=10 * 1024 ** 2, ignore=frozenset(), pickle_protocol=PICKLE_PROTOCOL):
    """Save the object to a file. Used internally by :meth:`gensim.utils.SaveLoad.save()`.

        Parameters
        ----------
        fname : str
            Path to file.
        separately : list, optional
            Iterable of attributes than need to store distinctly.
        sep_limit : int, optional
            Limit for separation.
        ignore : frozenset, optional
            Attributes that shouldn't be store.
        pickle_protocol : int, optional
            Protocol number for pickle.

        Notes
        -----
        If `separately` is None, automatically detect large numpy/scipy.sparse arrays in the object being stored,
        and store them into separate files. This avoids pickle memory errors and allows mmap'ing large arrays back
        on load efficiently.

        You can also set `separately` manually, in which case it must be a list of attribute names to be stored
        in separate files. The automatic check is not performed in this case.

        """
    compress, subname = SaveLoad._adapt_by_suffix(fname)
    restores = self._save_specials(fname, separately, sep_limit, ignore, pickle_protocol, compress, subname)
    try:
        pickle(self, fname, protocol=pickle_protocol)
    finally:
        for obj, asides in restores:
            for attrib, val in asides.items():
                with ignore_deprecation_warning():
                    setattr(obj, attrib, val)
    logger.info('saved %s', fname)