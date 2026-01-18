import sys
import os
import time
import uuid
import shlex
import threading
import shutil
import subprocess
import logging
import inspect
import ctypes
import runpy
import requests
import psutil
import multiprocess
from dash.testing.errors import (
from dash.testing import wait
class KillerThread(threading.Thread):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._old_threads = list(threading._active.keys())

    def kill(self):
        for thread_id in list(threading._active):
            if thread_id in self._old_threads:
                continue
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
            if res == 0:
                raise ValueError(f'Invalid thread id: {thread_id}')
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
                raise SystemExit('Stopping thread failure')