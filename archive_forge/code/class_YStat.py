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
class YStat(dict):
    """
    Class to hold a profile result line in a dict object, which all items can also be accessed as
    instance attributes where their attribute name is the given key. Mimicked NamedTuples.
    """
    _KEYS = {}

    def __init__(self, values):
        super().__init__()
        for key, i in self._KEYS.items():
            setattr(self, key, values[i])

    def __setattr__(self, name, value):
        self[self._KEYS[name]] = value
        super().__setattr__(name, value)