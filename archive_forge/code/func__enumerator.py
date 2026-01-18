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
def _enumerator(self, stat_entry):
    tstat = self._STAT_CLASS(stat_entry)
    self.append(tstat)