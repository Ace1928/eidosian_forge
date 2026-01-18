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
def CreateScopeRBACRoleBinding(self, name, role, user, group, labels=None):
    """Creates an ScopeRBACRoleBinding resource from the GKEHub API.

    Args:
      name: the full Scoperbacrolebinding resource name.
      role: the role.
      user: the user.
      group: the group.
      labels: labels for the RBACRoleBinding resource.

    Returns:
      An ScopeRBACRoleBinding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
      calliope_exceptions.RequiredArgumentException: if a required field is
        missing
    """
    rolebinding = self.messages.RBACRoleBinding(name=name, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())), user=user, group=group, labels=labels)
    resource = resources.REGISTRY.ParseRelativeName(name, 'gkehub.projects.locations.scopes.rbacrolebindings', api_version='v1alpha')
    req = self.messages.GkehubProjectsLocationsScopesRbacrolebindingsCreateRequest(rBACRoleBinding=rolebinding, rbacrolebindingId=resource.Name(), parent=resource.Parent().RelativeName())
    op = self.client.projects_locations_scopes_rbacrolebindings.Create(req)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_scopes_rbacrolebindings, self.client.projects_locations_operations), op_resource, 'Waiting for rbacrolebinding to be created')