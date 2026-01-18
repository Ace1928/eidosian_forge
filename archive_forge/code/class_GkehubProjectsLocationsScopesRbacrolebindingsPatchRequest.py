from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesRbacrolebindingsPatchRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesRbacrolebindingsPatchRequest object.

  Fields:
    name: The resource name for the rbacrolebinding `projects/{project}/locati
      ons/{location}/scopes/{scope}/rbacrolebindings/{rbacrolebinding}` or `pr
      ojects/{project}/locations/{location}/memberships/{membership}/rbacroleb
      indings/{rbacrolebinding}`
    rBACRoleBinding: A RBACRoleBinding resource to be passed as the request
      body.
    updateMask: Required. The fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    rBACRoleBinding = _messages.MessageField('RBACRoleBinding', 2)
    updateMask = _messages.StringField(3)