import asyncio
import os
import multiprocessing
import signal
import subprocess
import sys
import time
from binascii import hexlify
from tornado.concurrent import (
from tornado import ioloop
from tornado.iostream import PipeIOStream
from tornado.log import gen_log
import typing
from typing import Optional, Any, Callable
@classmethod
def _try_cleanup_process(cls, pid: int) -> None:
    try:
        ret_pid, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return
    if ret_pid == 0:
        return
    assert ret_pid == pid
    subproc = cls._waiting.pop(pid)
    subproc.io_loop.add_callback(subproc._set_returncode, status)