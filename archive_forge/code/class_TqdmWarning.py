import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet
from ._monitor import TMonitor
from .utils import (
class TqdmWarning(Warning):
    """base class for all tqdm warnings.

    Used for non-external-code-breaking errors, such as garbled printing.
    """

    def __init__(self, msg, fp_write=None, *a, **k):
        if fp_write is not None:
            fp_write('\n' + self.__class__.__name__ + ': ' + str(msg).rstrip() + '\n')
        else:
            super(TqdmWarning, self).__init__(msg, *a, **k)