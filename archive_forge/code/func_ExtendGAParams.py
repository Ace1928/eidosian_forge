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
def ExtendGAParams(self, new_params):
    """Extends self.ga_params to include new parameters.

    This is only used to record parameters that are sent with every event type,
    such as global and command-level options.

    Args:
      new_params: A dictionary of key-value parameters to send.
    """
    self.ga_params.update(new_params)