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
def DeleteMembershipRbacRoleBinding(self, name):
    """Deletes a Membership RBAC RoleBinding resource.

    Args:
      name: the resource name of the Membership RBAC RoleBinding.

    Returns:
      A long running operation for the deletion.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error.
    """
    req = self.messages.GkehubProjectsLocationsMembershipsRbacrolebindingsDeleteRequest(name=name)
    return self.client.projects_locations_memberships_rbacrolebindings.Delete(req)