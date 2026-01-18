import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
def _guard_kind(kind):
    """Guard to ensure that an event kind is valid.

    All event kinds with a "numba:" prefix must be defined in the pre-defined
    ``numba.core.event._builtin_kinds``.
    Custom event kinds are allowed by not using the above prefix.

    Parameters
    ----------
    kind : str

    Return
    ------
    res : str
    """
    if kind.startswith('numba:') and kind not in _builtin_kinds:
        msg = f"{kind} is not a valid event kind, it starts with the reserved prefix 'numba:'"
        raise ValueError(msg)
    return kind