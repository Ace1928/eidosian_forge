import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
class AsyncLoggingHandler:

    def __init__(self, logfile=None, maxsize=1024):
        _queue = OverflowingQueue(maxsize)
        if logfile is None:
            _handler = logging.StreamHandler()
        else:
            _handler = logging.FileHandler(logfile)
        self._listener = logging.handlers.QueueListener(_queue, _handler)
        self._async_handler = logging.handlers.QueueHandler(_queue)
        _handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S'))

    def __enter__(self):
        self._listener.start()
        return self._async_handler

    def __exit__(self, exc_type, exc_value, traceback):
        self._listener.stop()