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
class ProcINotify:
    """
    Access (read, write) inotify's variables through /proc/sys/. Note that
    usually it requires administrator rights to update them.

    Examples:
      - Read max_queued_events attribute: myvar = max_queued_events.value
      - Update max_queued_events attribute: max_queued_events.value = 42
    """

    def __init__(self, attr):
        self._base = '/proc/sys/fs/inotify'
        self._attr = attr

    def get_val(self):
        """
        Gets attribute's value.

        @return: stored value.
        @rtype: int
        @raise IOError: if corresponding file in /proc/sys cannot be read.
        """
        with open(os.path.join(self._base, self._attr), 'r') as file_obj:
            return int(file_obj.readline())

    def set_val(self, nval):
        """
        Sets new attribute's value.

        @param nval: replaces current value by nval.
        @type nval: int
        @raise IOError: if corresponding file in /proc/sys cannot be written.
        """
        with open(os.path.join(self._base, self._attr), 'w') as file_obj:
            file_obj.write(str(nval) + '\n')
    value = property(get_val, set_val)

    def __repr__(self):
        return '<%s=%d>' % (self._attr, self.get_val())