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
def CreateRBACRoleBinding(self, name, role, user, group):
    """Creates an RBACRoleBinding resource from the GKEHub API.

    Args:
      name: the full rbacrolebinding resource name.
      role: the role.
      user: the user.
      group: the group.

    Returns:
      An RBACRoleBinding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
      calliope_exceptions.RequiredArgumentException: if a required field is
        missing
    """
    rolebinding = self.messages.RBACRoleBinding(name=name, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())), user=user, group=group)
    resource = resources.REGISTRY.ParseRelativeName(name, 'gkehub.projects.locations.namespaces.rbacrolebindings', api_version='v1alpha')
    req = self.messages.GkehubProjectsLocationsNamespacesRbacrolebindingsCreateRequest(rBACRoleBinding=rolebinding, rbacrolebindingId=resource.Name(), parent=resource.Parent().RelativeName())
    return self.client.projects_locations_namespaces_rbacrolebindings.Create(req)