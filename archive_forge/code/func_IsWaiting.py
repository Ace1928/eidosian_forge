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
def IsWaiting(self, stage):
    """Returns True if the stage is not yet started."""
    stage = self._ValidateStage(stage, allow_complete=True)
    return stage.status == StageCompletionStatus.NOT_STARTED