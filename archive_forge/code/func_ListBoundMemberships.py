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
def ListBoundMemberships(self, scope_path):
    """Lists memberships bound to a scope.

    Args:
      scope_path: Full resource path of the scope for listing bound memberships.

    Returns:
      A ListMembershipsResponse (list of memberships and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    req = self.messages.GkehubProjectsLocationsScopesListMembershipsRequest(pageToken='', scopeName=scope_path)
    return list_pager.YieldFromList(self.client.projects_locations_scopes, req, method='ListMemberships', field='memberships', batch_size_attribute=None)