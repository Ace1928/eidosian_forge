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
class _StubProgressTracker(NoOpProgressTracker):
    """A Progress tracker that only prints deterministic start and end points.

  No UX about tracking should be exposed here. This is strictly for being able
  to tell that the tracker ran, not what it actually looks like.
  """

    def __init__(self, message, interruptable, aborted_message):
        super(_StubProgressTracker, self).__init__(interruptable, aborted_message)
        self._message = message or ''
        self._stream = sys.stderr

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val:
            status = 'SUCCESS'
        elif isinstance(exc_val, console_io.OperationCancelledError):
            status = 'INTERRUPTED'
        else:
            status = 'FAILURE'
        if log.IsUserOutputEnabled():
            self._stream.write(console_io.JsonUXStub(console_io.UXElementType.PROGRESS_TRACKER, message=self._message, status=status) + '\n')
        return super(_StubProgressTracker, self).__exit__(exc_type, exc_val, exc_tb)