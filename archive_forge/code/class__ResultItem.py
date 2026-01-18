import os
from concurrent.futures import _base
import queue
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
from traceback import format_exception
class _ResultItem(object):

    def __init__(self, work_id, exception=None, result=None, exit_pid=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result
        self.exit_pid = exit_pid