from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def SetUserOutputEnabled(self, enabled):
    """Sets whether user output should go to the console.

    Args:
      enabled: bool, True to enable output, False to suppress.  If None, the
        value from properties or the default will be used.

    Returns:
      bool, The old value of enabled.
    """
    if enabled is None:
        enabled = properties.VALUES.core.user_output_enabled.GetBool(validate=False)
    if enabled is None:
        enabled = DEFAULT_USER_OUTPUT_ENABLED
    self._user_output_filter.enabled = enabled
    old_enabled = self.user_output_enabled
    self.user_output_enabled = enabled
    return old_enabled