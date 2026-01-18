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
def _get_greenlet_name():
    return getcurrent().__class__.__name__