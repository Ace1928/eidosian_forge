import collections
import collections.abc
import concurrent.futures
import errno
import functools
import heapq
import itertools
import os
import socket
import stat
import subprocess
import threading
import time
import traceback
import sys
import warnings
import weakref
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import protocols
from . import sslproto
from . import staggered
from . import tasks
from . import transports
from . import trsock
from .log import logger
def _check_sendfile_params(self, sock, file, offset, count):
    if 'b' not in getattr(file, 'mode', 'b'):
        raise ValueError('file should be opened in binary mode')
    if not sock.type == socket.SOCK_STREAM:
        raise ValueError('only SOCK_STREAM type sockets are supported')
    if count is not None:
        if not isinstance(count, int):
            raise TypeError('count must be a positive integer (got {!r})'.format(count))
        if count <= 0:
            raise ValueError('count must be a positive integer (got {!r})'.format(count))
    if not isinstance(offset, int):
        raise TypeError('offset must be a non-negative integer (got {!r})'.format(offset))
    if offset < 0:
        raise ValueError('offset must be a non-negative integer (got {!r})'.format(offset))