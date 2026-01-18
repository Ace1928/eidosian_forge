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
def _GenerateStagePrefix(self, stage_status, tick_mark):
    if stage_status == StageCompletionStatus.NOT_STARTED:
        tick_mark = self._symbols.not_started
    elif stage_status == StageCompletionStatus.SUCCESS:
        tick_mark = self._symbols.success
    elif stage_status == StageCompletionStatus.FAILED:
        tick_mark = self._symbols.failed
    elif stage_status == StageCompletionStatus.INTERRUPTED:
        tick_mark = self._symbols.interrupted
    return tick_mark + ' ' * (self._symbols.prefix_length - len(tick_mark))