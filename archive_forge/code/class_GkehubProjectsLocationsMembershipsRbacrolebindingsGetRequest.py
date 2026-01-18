from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsRbacrolebindingsGetRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsRbacrolebindingsGetRequest object.

  Fields:
    name: Required. The RBACRoleBinding resource name in the format
      `projects/*/locations/*/memberships/*/rbacrolebindings/*`.
  """
    name = _messages.StringField(1, required=True)