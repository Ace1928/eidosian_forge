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
@CaptureAndLogException
def LogPerformanceSummaryParams(**params_to_update):
    """Logs parameters necessary for a PerformanceSummary.

  gsutil periodically monitors its own threads; at the end of the execution of
  each cp/rsync command, it will present a performance summary of the command
  run.

  Args:
    **params_to_update: A dictionary. See UpdatePerformanceSummaryParams for
        details. The argument ambiguity at this level allows for flexibility in
        dealing with arguments that are processed similarly.
  """
    collector = MetricsCollector.GetCollector()
    if collector:
        collector.UpdatePerformanceSummaryParams(params_to_update)