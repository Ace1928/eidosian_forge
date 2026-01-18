from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesServicePerimetersCreateRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesServicePerimetersCreateRequest
  object.

  Fields:
    parent: Required. Resource name for the access policy which owns this
      Service Perimeter. Format: `accessPolicies/{policy_id}`
    servicePerimeter: A ServicePerimeter resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    servicePerimeter = _messages.MessageField('ServicePerimeter', 2)