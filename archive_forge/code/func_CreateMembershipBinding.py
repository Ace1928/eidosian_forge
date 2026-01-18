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
def CreateMembershipBinding(self, name, scope, labels=None):
    """Creates a Membership-Binding resource from the GKEHub API.

    Args:
      name: the full binding resource name.
      scope: the Scope to be associated with Binding.
      labels: labels for the membership binding resource

    Returns:
      A Membership-Binding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
      calliope_exceptions.RequiredArgumentException: if a required field is
        missing
    """
    binding = self.messages.MembershipBinding(name=name, scope=scope, labels=labels)
    resource = resources.REGISTRY.ParseRelativeName(name, 'gkehub.projects.locations.memberships.bindings', api_version='v1alpha')
    req = self.messages.GkehubProjectsLocationsMembershipsBindingsCreateRequest(membershipBinding=binding, membershipBindingId=resource.Name(), parent=resource.Parent().RelativeName())
    op = self.client.projects_locations_memberships_bindings.Create(req)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_memberships_bindings, self.client.projects_locations_operations), op_resource, 'Waiting for membership binding to be created')