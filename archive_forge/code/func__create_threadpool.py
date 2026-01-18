import asyncio
import concurrent.futures
import errno
import os
import sys
import socket
import ssl
import stat
from tornado.concurrent import dummy_executor, run_on_executor
from tornado.ioloop import IOLoop
from tornado.util import Configurable, errno_from_exception
from typing import List, Callable, Any, Type, Dict, Union, Tuple, Awaitable, Optional
@classmethod
def _create_threadpool(cls, num_threads: int) -> concurrent.futures.ThreadPoolExecutor:
    pid = os.getpid()
    if cls._threadpool_pid != pid:
        cls._threadpool = None
    if cls._threadpool is None:
        cls._threadpool = concurrent.futures.ThreadPoolExecutor(num_threads)
        cls._threadpool_pid = pid
    return cls._threadpool