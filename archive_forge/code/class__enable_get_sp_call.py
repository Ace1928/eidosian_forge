from functools import wraps
import weakref
import abc
import warnings
from ..data_sparsifier import BaseDataSparsifier
class _enable_get_sp_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_sp_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_sp_called_within_step = False