from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def ManagedByFromString(managed_by):
    """Parses a string into a MANAGED_BY enum.

  MANAGED_BY is an enum of who manages a service account key resource. IAM
  will rotate any SYSTEM_MANAGED keys by default.

  Args:
    managed_by: A string representation of a MANAGED_BY. Can be one of *user*,
      *system* or *any*.

  Returns:
    A KeyTypeValueValuesEnum (MANAGED_BY) value.
  """
    if managed_by == 'user':
        return [MANAGED_BY.USER_MANAGED]
    elif managed_by == 'system':
        return [MANAGED_BY.SYSTEM_MANAGED]
    elif managed_by == 'any':
        return []
    else:
        return [MANAGED_BY.KEY_TYPE_UNSPECIFIED]