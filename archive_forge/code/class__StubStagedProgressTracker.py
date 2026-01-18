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
class _StubStagedProgressTracker(NoOpStagedProgressTracker):
    """Staged tracker that only prints deterministic start and end points.

  No UX about tracking should be exposed here. This is strictly for being able
  to tell that the tracker ran, not what it actually looks like.
  """

    def __init__(self, message, stages, interruptable, aborted_message):
        super(_StubStagedProgressTracker, self).__init__(stages, interruptable, aborted_message)
        self._message = message
        self._succeeded_stages = []
        self._failed_stage = None
        self._stream = sys.stderr

    def _CompleteStage(self, stage):
        self._succeeded_stages.append(stage.header)

    def _FailStage(self, stage, exception, message=None):
        self._failed_stage = stage.header
        raise exception

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val and isinstance(exc_val, console_io.OperationCancelledError):
            status_message = 'INTERRUPTED'
        elif exc_val:
            status_message = 'FAILURE'
        elif self.HasWarning():
            status_message = 'WARNING'
        else:
            status_message = 'SUCCESS'
        if log.IsUserOutputEnabled():
            self._stream.write(console_io.JsonUXStub(console_io.UXElementType.STAGED_PROGRESS_TRACKER, message=self._message, status=status_message, succeeded_stages=self._succeeded_stages, failed_stage=self._failed_stage) + '\n')
        return super(_StubStagedProgressTracker, self).__exit__(exc_type, exc_val, exc_tb)