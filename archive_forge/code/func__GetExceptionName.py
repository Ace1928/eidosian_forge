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
def _GetExceptionName(cls):
    """Returns the exception name used as index into _KNOWN_ERRORS from type."""
    return cls.__module__ + '.' + cls.__name__