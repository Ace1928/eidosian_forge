import time as _time
from sys import exit as sysexit
from os import _exit as osexit
from threading import Thread, Semaphore
from multiprocessing import Process, cpu_count
def _run_via_pool(*args, **kwargs):
    with config['POOLS'][config['POOL_NAME']]['pool']:
        return callee(*args, **kwargs)