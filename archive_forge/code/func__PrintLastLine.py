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
def _PrintLastLine(self, msg=''):
    """Print a pretty ending line to identify end of build output logs."""
    width, _ = console_attr_os.GetTermSize()
    self._PrintLogLine(msg.center(width, self.OUTPUT_LINE_CHAR) + '\n')