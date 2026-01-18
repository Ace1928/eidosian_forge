from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddRolesResponse(_messages.Message):
    """Represents IAM roles added to the shared VPC host project.

  Fields:
    policyBinding: Required. List of policy bindings that were added to the
      shared VPC host project.
  """
    policyBinding = _messages.MessageField('PolicyBinding', 1, repeated=True)