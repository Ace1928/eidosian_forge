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
class StageCompletionStatus(enum.Enum):
    """Indicates the completion status of a stage."""
    NOT_STARTED = 'not started'
    RUNNING = 'still running'
    SUCCESS = 'done'
    FAILED = 'failed'
    INTERRUPTED = 'interrupted'
    WARNING = 'warning'