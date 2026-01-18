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
def _CollectPerformanceSummaryMetric(self):
    """Aggregates PerformanceSummary info and adds the metric to the list."""
    if self.perf_sum_params is None:
        return
    custom_params = {}
    for attr_name, label in (('num_processes', 'Num Processes'), ('num_threads', 'Num Threads'), ('num_retryable_service_errors', 'Num Retryable Service Errors'), ('num_retryable_network_errors', 'Num Retryable Network Errors'), ('avg_throughput', 'Average Overall Throughput'), ('num_objects_transferred', 'Number of Files/Objects Transferred'), ('total_bytes_transferred', 'Size of Files/Objects Transferred')):
        custom_params[_GA_LABEL_MAP[label]] = getattr(self.perf_sum_params, attr_name)
    if system_util.IS_LINUX:
        disk_start = self.perf_sum_params.disk_counters_start
        disk_end = system_util.GetDiskCounters()
        custom_params[_GA_LABEL_MAP['Disk I/O Time']] = sum([stat[4] + stat[5] for stat in disk_end.values()]) - sum([stat[4] + stat[5] for stat in disk_start.values()])
    if self.perf_sum_params.has_cloud_src:
        src_url_type = 'both' if self.perf_sum_params.has_file_src else 'cloud'
    else:
        src_url_type = 'file'
    custom_params[_GA_LABEL_MAP['Source URL Type']] = src_url_type
    if self.perf_sum_params.uses_fan:
        strategy = 'both' if self.perf_sum_params.uses_slice else 'fan'
    else:
        strategy = 'slice' if self.perf_sum_params.uses_slice else 'none'
    custom_params[_GA_LABEL_MAP['Parallelism Strategy']] = strategy
    total_time = self.perf_sum_params.thread_idle_time + self.perf_sum_params.thread_execution_time
    if total_time:
        custom_params[_GA_LABEL_MAP['Thread Idle Time Percent']] = float(self.perf_sum_params.thread_idle_time) / float(total_time)
    if self.perf_sum_params.thread_throughputs:
        throughputs = [thread.GetThroughput() for thread in self.perf_sum_params.thread_throughputs.values()]
        custom_params[_GA_LABEL_MAP['Slowest Thread Throughput']] = min(throughputs)
        custom_params[_GA_LABEL_MAP['Fastest Thread Throughput']] = max(throughputs)
    custom_params[_GA_LABEL_MAP['Provider Types']] = ','.join(sorted(self.perf_sum_params.provider_types))
    transfer_types = {'CloudToCloud': self.perf_sum_params.has_cloud_src and self.perf_sum_params.has_cloud_dst, 'CloudToFile': self.perf_sum_params.has_cloud_src and self.perf_sum_params.has_file_dst, 'DaisyChain': self.perf_sum_params.is_daisy_chain, 'FileToCloud': self.perf_sum_params.has_file_src and self.perf_sum_params.has_cloud_dst, 'FileToFile': self.perf_sum_params.has_file_src and self.perf_sum_params.has_file_dst}
    action = ','.join(sorted([transfer_type for transfer_type, cond in six.iteritems(transfer_types) if cond]))
    apply_execution_time = _GetTimeInMillis(self.perf_sum_params.total_elapsed_time)
    self.CollectGAMetric(category=_GA_PERFSUM_CATEGORY, action=action, execution_time=apply_execution_time, **custom_params)