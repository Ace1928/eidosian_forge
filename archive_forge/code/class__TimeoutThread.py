from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import os.path
import subprocess
import threading
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
class _TimeoutThread(object):
    """A context manager based on threading.Timer.

  Pass a function to call after the given time has passed. If you exit before
  the timer fires, nothing happens. If you exit after we've had to call the
  timer function, we raise TimeoutError at exit time.
  """

    def __init__(self, func, timeout_sec, error_format='Task ran for more than {timeout_sec} seconds'):
        self.func = func
        self.timeout_sec = timeout_sec
        self.error_format = error_format
        self.timer = None

    def __enter__(self):
        self.Reset()
        return self

    def Reset(self):
        if self.timer is not None:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout_sec, self.func)
        self.timer.start()

    def __exit__(self, exc_type, exc_value, traceback):
        timed_out = self.timer.finished.is_set()
        self.timer.cancel()
        if timed_out:
            raise utils.TimeoutError(self.error_format.format(timeout_sec=self.timeout_sec))