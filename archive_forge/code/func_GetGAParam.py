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
def GetGAParam(self, param_name):
    """Convenience function for getting a ga_param of the collector.

    Args:
      param_name: The descriptive name of the param (e.g. 'Command Name'). Must
        be a key in _GA_LABEL_MAP.

    Returns:
      The GA parameter specified, or None.
    """
    return self.ga_params.get(_GA_LABEL_MAP[param_name])