import ctypes
import struct
from eventlet import patcher
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
class IOQueue(Queue.Queue, object):

    def __init__(self, client_connected):
        Queue.Queue.__init__(self)
        self._client_connected = client_connected

    def get(self, timeout=IO_QUEUE_TIMEOUT, continue_on_timeout=True):
        while self._client_connected.isSet():
            try:
                return Queue.Queue.get(self, timeout=timeout)
            except Queue.Empty:
                if continue_on_timeout:
                    continue
                else:
                    break

    def put(self, item, timeout=IO_QUEUE_TIMEOUT):
        while self._client_connected.isSet():
            try:
                return Queue.Queue.put(self, item, timeout=timeout)
            except Queue.Full:
                continue

    def get_burst(self, timeout=IO_QUEUE_TIMEOUT, burst_timeout=IO_QUEUE_BURST_TIMEOUT, max_size=constants.SERIAL_CONSOLE_BUFFER_SIZE):
        data = self.get(timeout=timeout)
        while data and len(data) <= max_size:
            chunk = self.get(timeout=burst_timeout, continue_on_timeout=False)
            if chunk:
                data += chunk
            else:
                break
        return data