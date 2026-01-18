from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
from collections import defaultdict
from functools import wraps
import logging
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import six
from six.moves import input
from six.moves import urllib
import boto
from gslib import VERSION
from gslib.metrics_tuple import Metric
from gslib.utils import system_util
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import HumanReadableToBytes
def _ProcessFileMessage(self, file_message):
    """Processes FileMessages for thread throughput calculations.

    Update a thread's throughput based on the FileMessage, which marks the start
    or end of a file or component transfer. The FileMessage provides the number
    of bytes transferred as well as start and end time.

    Args:
      file_message: The FileMessage to process.
    """
    thread_info = self.perf_sum_params.thread_throughputs[file_message.process_id, file_message.thread_id]
    if file_message.finished:
        if not (self.perf_sum_params.uses_slice or self.perf_sum_params.uses_fan):
            self.perf_sum_params.num_objects_transferred += 1
        thread_info.LogTaskEnd(file_message.time)
    else:
        thread_info.LogTaskStart(file_message.time, file_message.size)