from torch.ao.pruning import BaseSparsifier
from functools import wraps
import warnings
import weakref
class _enable_get_sl_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_sl_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_sl_called_within_step = False