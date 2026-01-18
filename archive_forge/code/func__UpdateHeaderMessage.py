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
def _UpdateHeaderMessage(self, prefix):
    message = prefix + self._message
    if self._header_stage.message:
        message += ' ' + self._header_stage.message
    self._UpdateMessage(self._header_message, message)