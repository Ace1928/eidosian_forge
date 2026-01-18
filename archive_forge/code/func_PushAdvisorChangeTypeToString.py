from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def PushAdvisorChangeTypeToString(change_type):
    """Convert a ConfigChange.ChangeType enum to a string.

  Args:
    change_type: The ConfigChange.ChangeType enum to convert.

  Returns:
    An easily readable string representing the ConfigChange.ChangeType enum.
  """
    messages = GetMessagesModule()
    enums = messages.ConfigChange.ChangeTypeValueValuesEnum
    if change_type in [enums.ADDED, enums.REMOVED, enums.MODIFIED]:
        return six.text_type(change_type).lower()
    else:
        return '[unknown]'