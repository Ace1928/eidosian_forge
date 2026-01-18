import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
def _exit_wrapper(exc_type, exc, tb):
    callback(*args, **kwds)