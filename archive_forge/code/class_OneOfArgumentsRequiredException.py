from within calliope.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from functools import wraps
import os
import sys
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
import six
class OneOfArgumentsRequiredException(ToolException):
    """An exception for when one of usually optional arguments is required.
  """

    def __init__(self, parameters, message):
        super(OneOfArgumentsRequiredException, self).__init__('One of arguments [{0}] is required: {1}'.format(', '.join(parameters), message))
        self.parameters = parameters