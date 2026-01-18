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
def Wrap(func):
    """Wrapper function for the decorator."""

    @wraps(func)
    def TryFunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except error_types:
            core_exceptions.reraise(NewErrorFromCurrentException(error))
    return TryFunc