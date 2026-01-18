from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RBACRoleBindingActuationFeatureSpec(_messages.Message):
    """**RBAC RoleBinding Actuation**: The Hub-wide input for the
  RBACRoleBindingActuation feature.

  Fields:
    actuationDisabled: A boolean attribute.
  """
    actuationDisabled = _messages.BooleanField(1)