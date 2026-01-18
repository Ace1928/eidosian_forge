from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
def ListMembershipBindings(self, project, membership, location='global'):
    """Lists Bindings in a Membership.

    Args:
      project: the project containing the Membership to list Bindings from.
      membership: the Membership to list Bindings from.
      location: the Membrship location to list Bindings

    Returns:
      A ListMembershipBindingResponse (list of bindings and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    req = self.messages.GkehubProjectsLocationsMembershipsBindingsListRequest(pageToken='', parent=util.MembershipBindingParentName(project, membership, location, self.release_track))
    return list_pager.YieldFromList(self.client.projects_locations_memberships_bindings, req, field='membershipBindings', batch_size_attribute=None)