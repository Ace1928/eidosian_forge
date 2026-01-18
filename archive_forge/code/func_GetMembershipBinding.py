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
def GetMembershipBinding(self, name):
    """Gets a Membership-Binding resource from the GKEHub API.

    Args:
      name: the full membership-binding resource name.

    Returns:
      A Membership-Binding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    req = self.messages.GkehubProjectsLocationsMembershipsBindingsGetRequest(name=name)
    return self.client.projects_locations_memberships_bindings.Get(req)