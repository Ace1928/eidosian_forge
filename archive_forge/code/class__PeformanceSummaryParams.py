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
class _PeformanceSummaryParams(object):
    """This class contains information to create a PerformanceSummary event."""

    def __init__(self):
        self.num_processes = 0
        self.num_threads = 0
        self.num_retryable_service_errors = 0
        self.num_retryable_network_errors = 0
        self.provider_types = set()
        if system_util.IS_LINUX:
            self.disk_counters_start = system_util.GetDiskCounters()
        self.uses_fan = False
        self.uses_slice = False
        self.thread_idle_time = 0
        self.thread_execution_time = 0
        self.thread_throughputs = defaultdict(self._ThreadThroughputInformation)
        self.avg_throughput = None
        self.total_elapsed_time = None
        self.total_bytes_transferred = None
        self.num_objects_transferred = 0
        self.is_daisy_chain = False
        self.has_file_dst = False
        self.has_cloud_dst = False
        self.has_file_src = False
        self.has_cloud_src = False

    class _ThreadThroughputInformation(object):
        """A class to keep track of throughput information for a single thread."""

        def __init__(self):
            self.total_bytes_transferred = 0
            self.total_elapsed_time = 0
            self.task_start_time = None
            self.task_size = None

        def LogTaskStart(self, start_time, bytes_to_transfer):
            self.task_start_time = start_time
            self.task_size = bytes_to_transfer

        def LogTaskEnd(self, end_time):
            self.total_elapsed_time += end_time - self.task_start_time
            self.total_bytes_transferred += self.task_size
            self.task_start_time = None
            self.task_size = None

        def GetThroughput(self):
            return CalculateThroughput(self.total_bytes_transferred, self.total_elapsed_time)