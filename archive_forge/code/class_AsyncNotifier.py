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
class AsyncNotifier(asyncore.file_dispatcher, Notifier):
    """
    This notifier inherits from asyncore.file_dispatcher in order to be able to
    use pyinotify along with the asyncore framework.

    """

    def __init__(self, watch_manager, default_proc_fun=None, read_freq=0, threshold=0, timeout=None, channel_map=None):
        """
        Initializes the async notifier. The only additional parameter is
        'channel_map' which is the optional asyncore private map. See
        Notifier class for the meaning of the others parameters.

        """
        Notifier.__init__(self, watch_manager, default_proc_fun, read_freq, threshold, timeout)
        asyncore.file_dispatcher.__init__(self, self._fd, channel_map)

    def handle_read(self):
        """
        When asyncore tells us we can read from the fd, we proceed processing
        events. This method can be overridden for handling a notification
        differently.

        """
        self.read_events()
        self.process_events()