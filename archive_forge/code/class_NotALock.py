import zmq
import logging
from itertools import chain
from bisect import bisect
import socket
from operator import add
from time import sleep, time
from toolz import accumulate, topk, pluck, merge, keymap
import uuid
from collections import defaultdict
from contextlib import contextmanager, suppress
from threading import Thread, Lock
from datetime import datetime
from multiprocessing import Process
import traceback
import sys
from .dict import Dict
from .file import File
from .buffer import Buffer
from . import core
from .core import Interface
from .file import File
class NotALock:

    def acquire(self):
        pass

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass