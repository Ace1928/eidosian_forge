import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def _default_infer_from_sig_threadaware(fn, *args, **kwargs):
    thrid = threading.get_ident()
    return _inferrers_threadaware.get(thrid, _inferrer_global)(fn, *args, **kwargs)