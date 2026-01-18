from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRBACRoleBindingsResponse(_messages.Message):
    """List of RBACRoleBindings.

  Fields:
    nextPageToken: A token to request the next page of resources from the
      `ListRBACRoleBindings` method. The value of an empty string means that
      there are no more resources to return.
    rbacrolebindings: The list of RBACRoleBindings
  """
    nextPageToken = _messages.StringField(1)
    rbacrolebindings = _messages.MessageField('RBACRoleBinding', 2, repeated=True)