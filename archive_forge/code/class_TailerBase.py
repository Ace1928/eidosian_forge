from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import re
import threading
import time
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import requests as creds_requests
from googlecloudsdk.core.util import encoding
import requests
class TailerBase(object):
    """Base class for log tailer classes."""
    LOG_OUTPUT_BEGIN = ' REMOTE BUILD OUTPUT '
    OUTPUT_LINE_CHAR = '-'

    def _ValidateScreenReader(self, text):
        """Modify output for better screen reader experience."""
        screen_reader = properties.VALUES.accessibility.screen_reader.GetBool()
        if screen_reader:
            return re.sub('---> ', '', text)
        return text

    def _PrintLogLine(self, text):
        """Testing Hook: This method enables better verification of output."""
        if self.out and text:
            self.out.Print(text.rstrip())

    def _PrintFirstLine(self, msg=LOG_OUTPUT_BEGIN):
        """Print a pretty starting line to identify start of build output logs."""
        width, _ = console_attr_os.GetTermSize()
        self._PrintLogLine(msg.center(width, self.OUTPUT_LINE_CHAR))

    def _PrintLastLine(self, msg=''):
        """Print a pretty ending line to identify end of build output logs."""
        width, _ = console_attr_os.GetTermSize()
        self._PrintLogLine(msg.center(width, self.OUTPUT_LINE_CHAR) + '\n')