import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class reset_terminal(object):

    def __init__(self):
        self._buf = bytearray(b'')

    def __enter__(self):
        return self._buf

    def __exit__(self, *exc_detail):
        self._buf += bytearray(Style.RESET_ALL.encode('utf-8'))