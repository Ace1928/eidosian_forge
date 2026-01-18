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
def _debug_check_sanity(self):
    """
        Check for basic sanity errors in stats. e.g: Check for duplicate stats.
        """
    for x in self:
        if self.count(x) > 1:
            return False
    return True