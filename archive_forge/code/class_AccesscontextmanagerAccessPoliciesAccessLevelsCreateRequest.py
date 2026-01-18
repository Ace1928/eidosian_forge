from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAccessLevelsCreateRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAccessLevelsCreateRequest object.

  Fields:
    accessLevel: A AccessLevel resource to be passed as the request body.
    parent: Required. Resource name for the access policy which owns this
      Access Level. Format: `accessPolicies/{policy_id}`
  """
    accessLevel = _messages.MessageField('AccessLevel', 1)
    parent = _messages.StringField(2, required=True)