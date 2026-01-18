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
def _CollectCommandAndErrorMetrics(self):
    """Aggregates command and error info and adds them to the metrics list."""
    command_name = self.GetGAParam('Command Name')
    if command_name:
        self.CollectGAMetric(category=_GA_COMMANDS_CATEGORY, action=command_name, **{_GA_LABEL_MAP['Retryable Errors']: sum(self.retryable_errors.values())})
    for error_type, num_errors in six.iteritems(self.retryable_errors):
        self.CollectGAMetric(category=_GA_ERRORRETRY_CATEGORY, action=error_type, **{_GA_LABEL_MAP['Retryable Errors']: num_errors})
    fatal_error_type = self.GetGAParam('Fatal Error')
    if fatal_error_type:
        self.CollectGAMetric(category=_GA_ERRORFATAL_CATEGORY, action=fatal_error_type)