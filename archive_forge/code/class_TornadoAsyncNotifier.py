import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
class TornadoAsyncNotifier(Notifier):
    """
    Tornado ioloop adapter.

    """

    def __init__(self, watch_manager, ioloop, callback=None, default_proc_fun=None, read_freq=0, threshold=0, timeout=None, channel_map=None):
        """
        Note that if later you must call ioloop.close() be sure to let the
        default parameter to all_fds=False.

        See example tornado_notifier.py for an example using this notifier.

        @param ioloop: Tornado's IO loop.
        @type ioloop: tornado.ioloop.IOLoop instance.
        @param callback: Functor called at the end of each call to handle_read
                         (IOLoop's read handler). Expects to receive the
                         notifier object (self) as single parameter.
        @type callback: callable object or function
        """
        self.io_loop = ioloop
        self.handle_read_callback = callback
        Notifier.__init__(self, watch_manager, default_proc_fun, read_freq, threshold, timeout)
        ioloop.add_handler(self._fd, self.handle_read, ioloop.READ)

    def stop(self):
        self.io_loop.remove_handler(self._fd)
        Notifier.stop(self)

    def handle_read(self, *args, **kwargs):
        """
        See comment in AsyncNotifier.

        """
        self.read_events()
        self.process_events()
        if self.handle_read_callback is not None:
            self.handle_read_callback(self)