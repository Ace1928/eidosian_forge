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
class YThreadStats(_YContextStats):
    _BACKEND = NATIVE_THREAD
    _STAT_CLASS = YThreadStat
    _SORT_TYPES = {'name': 0, 'id': 1, 'tid': 2, 'totaltime': 3, 'schedcount': 4, 'ttot': 3, 'scnt': 4}
    _DEFAULT_PRINT_COLUMNS = {0: ('name', 13), 1: ('id', 5), 2: ('tid', 15), 3: ('ttot', 8), 4: ('scnt', 10)}
    _ALL_COLUMNS = ['name', 'id', 'tid', 'ttot', 'scnt']