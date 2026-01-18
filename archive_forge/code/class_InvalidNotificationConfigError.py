from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import util
from googlecloudsdk.core import exceptions as core_exceptions
class InvalidNotificationConfigError(core_exceptions.Error):
    """Exception raised for errors in the input."""