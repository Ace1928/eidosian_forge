from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import os.path
import signal
import subprocess
import sys
import threading
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
class _SigInterruptedHandler(object):
    """Context manager to capture SIGINT and send it to a handler."""

    def __init__(self, handler):
        self._orig_handler = None
        self._handler = handler

    def __enter__(self):
        self._orig_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)

    def __exit__(self, exc_type, exc_value, tb):
        signal.signal(signal.SIGINT, self._orig_handler)