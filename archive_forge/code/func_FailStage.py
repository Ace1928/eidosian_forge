from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
import signal
import sys
import threading
import time
import enum
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import multiline
from googlecloudsdk.core.console.style import parser
import six
def FailStage(self, key, failure_exception, message=None):
    """Informs the progress tracker that this stage has failed.

    Args:
      key: str, key for the stage to fail.
      failure_exception: Exception, raised by __exit__.
      message: str, user visible message for failure.
    """
    stage = self._ValidateStage(key)
    with self._lock:
        stage.status = StageCompletionStatus.FAILED
        stage._is_done = True
        self._running_stages.discard(key)
        if message is not None:
            stage.message = message
        self._FailStage(stage, failure_exception, message)
    self.Tick()
    if failure_exception:
        self._PrintExitOutput(failed=True)
        self._exception_is_uncaught = False
        raise failure_exception