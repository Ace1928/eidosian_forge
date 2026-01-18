import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def _default_infer_from_sig(fn, *args, **kwargs):
    """This is the default backend dispatcher, used if no global backend has
    been set. Hot swapping this function out as below avoids having to check
    manually for a global backend or worse, a thread aware global backend, on
    every call to ``do``.
    """
    return _DISPATCHERS[fn](*args, **kwargs)