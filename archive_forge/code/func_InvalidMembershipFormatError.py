from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def InvalidMembershipFormatError(name):
    """Returns error for invalid membership resource names.

  Args:
    name: The membership resource name.

  Returns:
   An exceptions.Error for malformed membership names.
  """
    return exceptions.Error('Failed to get membership: {} does not match format projects/PROJECT_ID/locations/LOCATION/memberships/MEMBERSHIP_ID'.format(name))