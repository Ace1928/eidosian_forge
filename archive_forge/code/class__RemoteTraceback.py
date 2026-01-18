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
class _RemoteTraceback(Exception):

    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        return self.tb