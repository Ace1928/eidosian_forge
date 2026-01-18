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
def _save_as_YSTAT(self, path):
    with open(path, 'wb') as f:
        pickle.dump((self, self._clock_type), f, YPICKLE_PROTOCOL)