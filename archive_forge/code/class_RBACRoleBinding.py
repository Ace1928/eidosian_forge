from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class RBACRoleBinding(base.Group):
    """Fleet scope RBAC RoleBindings for permissions.

  This command group allows for manipulation of fleet namespace RBAC
  RoleBindings.

  ## EXAMPLES

  Manage RBAC RoleBindings:

    $ {command} --help
  """
    category = base.COMPUTE_CATEGORY