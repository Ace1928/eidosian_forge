import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def _save_as_PSTAT(self, path):
    """
        Save the profiling information as PSTAT.
        """
    _stats = convert2pstats(self)
    _stats.dump_stats(path)