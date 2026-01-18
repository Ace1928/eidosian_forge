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
class MinimumArgumentException(ToolException):
    """An exception for when one of several arguments is required."""

    def __init__(self, parameter_names, message=None):
        if message:
            message = ': {}'.format(message)
        else:
            message = ''
        super(MinimumArgumentException, self).__init__('One of [{0}] must be supplied{1}.'.format(', '.join(['{0}'.format(p) for p in parameter_names]), message))