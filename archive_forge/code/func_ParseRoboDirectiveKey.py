from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def ParseRoboDirectiveKey(key):
    """Returns a tuple representing a directive's type and resource name.

  Args:
    key: the directive key, which can be "<type>:<resource>" or "<resource>"

  Returns:
    A tuple of the directive's parsed type and resource name. If no type is
    specified, "text" will be returned as the default type.

  Raises:
    InvalidArgException: if the input format is incorrect or if the specified
    type is unsupported.
  """
    parts = key.split(':')
    resource_name = parts[-1]
    if len(parts) > 2:
        raise exceptions.InvalidArgException('robo_directives', 'Invalid format for key [{0}]. Use a colon only to separate action type and resource name.'.format(key))
    if len(parts) == 1:
        action_type = 'text'
    else:
        action_type = parts[0]
        supported_action_types = ['text', 'click', 'ignore']
        if action_type not in supported_action_types:
            raise exceptions.InvalidArgException('robo_directives', 'Unsupported action type [{0}]. Please choose one of [{1}]'.format(action_type, ', '.join(supported_action_types)))
    return (action_type, resource_name)