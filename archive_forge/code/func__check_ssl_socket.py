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
def _check_ssl_socket(sock):
    if ssl is not None and isinstance(sock, ssl.SSLSocket):
        raise TypeError('Socket cannot be of type SSLSocket')