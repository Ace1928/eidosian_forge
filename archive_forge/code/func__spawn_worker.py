import collections
import threading
import time
import socket
import warnings
import queue
from jaraco.functools import pass_none
def _spawn_worker(self):
    worker = WorkerThread(self.server)
    worker.name = 'CP Server {worker_name!s}'.format(worker_name=worker.name)
    worker.start()
    return worker