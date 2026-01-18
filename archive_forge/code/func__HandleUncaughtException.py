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
def _HandleUncaughtException(self, exc_value):
    if isinstance(exc_value, console_io.OperationCancelledError):
        self._Print('aborted by ctrl-c')
        self._PrintExitOutput(aborted=True)
    else:
        self._Print(self._GetStagedCompletedSuffix(StageCompletionStatus.FAILED))
        self._PrintExitOutput(failed=True)