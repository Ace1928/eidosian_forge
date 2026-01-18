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
@classmethod
def StartTestCollector(cls, endpoint='https://example.com', user_agent='user-agent-007', ga_params=None):
    """Reset the singleton MetricsCollector with testing parameters.

    Should only be used for tests, where we want to change the default
    parameters.

    Args:
      endpoint: str, URL to post to
      user_agent: str, User-Agent string for header.
      ga_params: A list of two-dimensional string tuples to send as parameters.
    """
    if cls.IsDisabled():
        os.environ['GSUTIL_TEST_ANALYTICS'] = '0'
    cls._disabled_cache = False
    cls._instance = cls(_GA_TID_TESTING, endpoint)
    if ga_params is None:
        ga_params = {'a': 'b', 'c': 'd'}
    cls._instance.ga_params = ga_params
    cls._instance.user_agent = user_agent
    if os.environ['GSUTIL_TEST_ANALYTICS'] != '2':
        cls._instance.start_time = 0