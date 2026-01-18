import unittest
import unittest.mock
import queue as pyqueue
import textwrap
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
import_helper.import_module('multiprocess.synchronize')
import threading
import multiprocess as multiprocessing
import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues
from multiprocess import util
from multiprocess.connection import wait
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
class ManagerMixin(BaseMixin):
    TYPE = 'manager'
    Process = multiprocessing.Process
    Queue = property(operator.attrgetter('manager.Queue'))
    JoinableQueue = property(operator.attrgetter('manager.JoinableQueue'))
    Lock = property(operator.attrgetter('manager.Lock'))
    RLock = property(operator.attrgetter('manager.RLock'))
    Semaphore = property(operator.attrgetter('manager.Semaphore'))
    BoundedSemaphore = property(operator.attrgetter('manager.BoundedSemaphore'))
    Condition = property(operator.attrgetter('manager.Condition'))
    Event = property(operator.attrgetter('manager.Event'))
    Barrier = property(operator.attrgetter('manager.Barrier'))
    Value = property(operator.attrgetter('manager.Value'))
    Array = property(operator.attrgetter('manager.Array'))
    list = property(operator.attrgetter('manager.list'))
    dict = property(operator.attrgetter('manager.dict'))
    Namespace = property(operator.attrgetter('manager.Namespace'))

    @classmethod
    def Pool(cls, *args, **kwds):
        return cls.manager.Pool(*args, **kwds)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.manager = multiprocessing.Manager()

    @classmethod
    def tearDownClass(cls):
        start_time = time.monotonic()
        t = 0.01
        while len(multiprocessing.active_children()) > 1:
            time.sleep(t)
            t *= 2
            dt = time.monotonic() - start_time
            if dt >= 5.0:
                test.support.environment_altered = True
                support.print_warning(f'multiprocess.Manager still has {multiprocessing.active_children()} active children after {dt} seconds')
                break
        gc.collect()
        if cls.manager._number_of_objects() != 0:
            test.support.environment_altered = True
            support.print_warning('Shared objects which still exist at manager shutdown:')
            support.print_warning(cls.manager._debug_info())
        cls.manager.shutdown()
        cls.manager.join()
        cls.manager = None
        super().tearDownClass()