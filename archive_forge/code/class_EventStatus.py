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
class EventStatus(enum.Enum):
    """Status of an event.
    """
    START = enum.auto()
    END = enum.auto()