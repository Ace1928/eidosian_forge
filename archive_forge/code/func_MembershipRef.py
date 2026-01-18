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
def MembershipRef(project, location, membership_id):
    """Get the resource name of a membership.

  Args:
    project: the project in which to create the membership
    location: the GCP region of the membership.
    membership_id: the ID of the membership.

  Returns:
    the full resource name of the membership in the format of
    `projects/{project}/locations/{location}/memberships/{membership_id}`
  """
    return '{}/memberships/{}'.format(ParentRef(project, location), membership_id)