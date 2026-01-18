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
@staticmethod
def _adapt_by_suffix(fname):
    """Get compress setting and filename for numpy file compression.

        Parameters
        ----------
        fname : str
            Input filename.

        Returns
        -------
        (bool, function)
            First argument will be True if `fname` compressed.

        """
    compress, suffix = (True, 'npz') if fname.endswith('.gz') or fname.endswith('.bz2') else (False, 'npy')
    return (compress, lambda *args: '.'.join(args + (suffix,)))