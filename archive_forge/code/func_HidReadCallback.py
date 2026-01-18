from six.moves import queue
from six.moves import range
import ctypes
import ctypes.util
import logging
import sys
import threading
from pyu2f import errors
from pyu2f.hid import base
def HidReadCallback(read_queue, result, sender, report_type, report_id, report, report_length):
    """Handles incoming IN report from HID device."""
    del result, sender, report_type, report_id
    incoming_bytes = [report[i] for i in range(report_length)]
    read_queue.put(incoming_bytes)