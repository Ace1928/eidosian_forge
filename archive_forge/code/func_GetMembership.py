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
def GetMembership(name, release_track=None):
    """Gets a Membership resource from the GKE Hub API.

  Args:
    name: the full resource name of the membership to get, e.g.,
      projects/foo/locations/global/memberships/name.
    release_track: the release_track used in the gcloud command, or None if it
      is not available.

  Returns:
    a Membership resource

  Raises:
    apitools.base.py.HttpError: if the request returns an HTTP error
    exceptions.Error: if the membership name is missing the ID or improperly
      formatted
  """
    if _MEMBERSHIP_RE.match(name) is None:
        raise InvalidMembershipFormatError(name)
    client = gkehub_api_util.GetApiClientForTrack(release_track)
    return client.projects_locations_memberships.Get(client.MESSAGES_MODULE.GkehubProjectsLocationsMembershipsGetRequest(name=name))