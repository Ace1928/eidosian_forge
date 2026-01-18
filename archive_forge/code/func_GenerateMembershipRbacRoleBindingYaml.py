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
def GenerateMembershipRbacRoleBindingYaml(self, name, role, user, group):
    """Gets YAML containing RBAC policies for a membership RBAC role binding.

    Args:
      name: the full Membership RBAC Role Binding resource name.
      role: the role for the RBAC policies.
      user: the user to apply the RBAC policies for.
      group: the group to apply the RBAC policies for.

    Returns:
      YAML text containing RBAC policies for a membership RBAC rolebinding.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error.
    """
    rolebinding = self.messages.RBACRoleBinding(name=name, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())), user=user, group=group)
    resource = resources.REGISTRY.ParseRelativeName(name, 'gkehub.projects.locations.memberships.rbacrolebindings', api_version='v1alpha')
    req = self.messages.GkehubProjectsLocationsMembershipsRbacrolebindingsGenerateMembershipRBACRoleBindingYAMLRequest(rBACRoleBinding=rolebinding, rbacrolebindingId=resource.Name(), parent=resource.Parent().RelativeName())
    return self.client.projects_locations_memberships_rbacrolebindings.GenerateMembershipRBACRoleBindingYAML(req)