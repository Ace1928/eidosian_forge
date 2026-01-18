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
def _update_logging_levels(self):
    """Updates absl logging levels to the current verbosity.

    Visibility: module-private
    """
    if not _absl_logger:
        return
    if self._value <= converter.ABSL_DEBUG:
        standard_verbosity = converter.absl_to_standard(self._value)
    else:
        standard_verbosity = logging.DEBUG - (self._value - 1)
    if _absl_handler in logging.root.handlers:
        _absl_logger.setLevel(logging.NOTSET)
        logging.root.setLevel(standard_verbosity)
    else:
        _absl_logger.setLevel(standard_verbosity)