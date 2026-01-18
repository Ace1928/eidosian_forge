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
def CompleteStage(self, key, message=None):
    """Informs the progress tracker that this stage has completed."""
    stage = self._ValidateStage(key)
    with self._lock:
        stage.status = StageCompletionStatus.SUCCESS
        stage._is_done = True
        self._running_stages.discard(key)
        if message is not None:
            stage.message = message
        self._CompleteStage(stage)
    self.Tick()