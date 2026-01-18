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
class _PStatHolder:

    def __init__(self, d):
        self.stats = d

    def create_stats(self):
        pass