from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import pickle
import re
import socket
import subprocess
import sys
import tempfile
import pprint
import six
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from boto.storage_uri import BucketStorageUri
from gslib import metrics
from gslib import VERSION
from gslib.cs_api_map import ApiSelector
import gslib.exception
from gslib.gcs_json_api import GcsJsonApi
from gslib.metrics import MetricsCollector
from gslib.metrics_tuple import Metric
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import HAS_S3_CREDS
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SkipForParFile
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.retry_util import LogAndHandleRetries
from gslib.utils.system_util import IS_LINUX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
from six import add_move, MovedModule
from six.moves import mock
def _GetAndCheckAllNumberMetrics(self, metrics_to_search, multithread=True):
    """Checks number metrics for PerformanceSummary tests.

    Args:
      metrics_to_search: The string of metrics to search.
      multithread: False if the the metrics were collected in a non-multithread
                   situation.

    Returns:
      (slowest_throughput, fastest_throughput, io_time) as floats.
    """

    def _ExtractNumberMetric(param_name):
        extracted_match = re.search(metrics._GA_LABEL_MAP[param_name] + '=(\\d+\\.?\\d*)&', metrics_to_search)
        if not extracted_match:
            self.fail('Could not find %s (%s) in metrics string %s' % (metrics._GA_LABEL_MAP[param_name], param_name, metrics_to_search))
        return float(extracted_match.group(1))
    if multithread:
        thread_idle_time = _ExtractNumberMetric('Thread Idle Time Percent')
        self.assertGreaterEqual(thread_idle_time, 0)
        self.assertLessEqual(thread_idle_time, 1)
    throughput = _ExtractNumberMetric('Average Overall Throughput')
    self.assertGreater(throughput, 0)
    slowest_throughput = _ExtractNumberMetric('Slowest Thread Throughput')
    fastest_throughput = _ExtractNumberMetric('Fastest Thread Throughput')
    self.assertGreaterEqual(fastest_throughput, slowest_throughput)
    self.assertGreater(slowest_throughput, 0)
    self.assertGreater(fastest_throughput, 0)
    io_time = None
    if IS_LINUX:
        io_time = _ExtractNumberMetric('Disk I/O Time')
        self.assertGreaterEqual(io_time, 0)
    return (slowest_throughput, fastest_throughput, io_time)