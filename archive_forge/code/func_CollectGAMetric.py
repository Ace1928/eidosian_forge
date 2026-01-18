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
def CollectGAMetric(self, category, action, label=VERSION, value=0, execution_time=None, **custom_params):
    """Adds a GA metric with the given parameters to the metrics queue.

    Args:
      category: str, the GA Event category.
      action: str, the GA Event action.
      label: str, the GA Event label.
      value: int, the GA Event value.
      execution_time: int, the execution time to record in ms.
      **custom_params: A dictionary of key, value pairs containing custom
          metrics and dimensions to send with the GA Event.
    """
    params = [('ec', category), ('ea', action), ('el', label), ('ev', value), (_GA_LABEL_MAP['Timestamp'], _GetTimeInMillis())]
    params.extend([(k, v) for k, v in six.iteritems(custom_params) if v is not None])
    params.extend([(k, v) for k, v in six.iteritems(self.ga_params) if v is not None])
    if execution_time is None:
        execution_time = _GetTimeInMillis() - self.start_time
    params.append((_GA_LABEL_MAP['Execution Time'], execution_time))
    data = urllib.parse.urlencode(sorted(params))
    self._metrics.append(Metric(endpoint=self.endpoint, method='POST', body=data, user_agent=self.user_agent))