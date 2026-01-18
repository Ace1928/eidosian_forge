from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesDeleteRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesDeleteRequest object.

  Fields:
    name: Required. Resource name for the access policy to delete. Format
      `accessPolicies/{policy_id}`
  """
    name = _messages.StringField(1, required=True)