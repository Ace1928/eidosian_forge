from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesServicePerimetersReplaceAllRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesServicePerimetersReplaceAllRequest
  object.

  Fields:
    parent: Required. Resource name for the access policy which owns these
      Service Perimeters. Format: `accessPolicies/{policy_id}`
    replaceServicePerimetersRequest: A ReplaceServicePerimetersRequest
      resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    replaceServicePerimetersRequest = _messages.MessageField('ReplaceServicePerimetersRequest', 2)