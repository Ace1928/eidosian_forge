import atexit
from ctypes import sizeof
import multiprocessing
import threading
import socket
import time
from cupyx.distributed import _klv_utils
from cupyx.distributed import _store_actions
def _set_process(self, process):
    self._process = process