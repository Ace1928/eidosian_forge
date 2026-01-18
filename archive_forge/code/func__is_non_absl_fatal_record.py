from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
def _is_non_absl_fatal_record(log_record):
    return log_record.levelno >= logging.FATAL and (not log_record.__dict__.get(_ABSL_LOG_FATAL, False))