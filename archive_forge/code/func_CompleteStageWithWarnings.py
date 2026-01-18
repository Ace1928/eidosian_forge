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
def CompleteStageWithWarnings(self, key, warning_messages):
    """Informs the progress tracker that this stage completed with warnings.

    Args:
      key: str, key for the stage to fail.
      warning_messages: list of str, user visible warning messages.
    """
    stage = self._ValidateStage(key)
    with self._lock:
        stage.status = StageCompletionStatus.WARNING
        stage._is_done = True
        self._running_stages.discard(key)
        self._exit_output_warnings.extend(warning_messages)
        self._completed_with_warnings_stages.append(stage.key)
        self._CompleteStageWithWarnings(stage, warning_messages)
    self.Tick()