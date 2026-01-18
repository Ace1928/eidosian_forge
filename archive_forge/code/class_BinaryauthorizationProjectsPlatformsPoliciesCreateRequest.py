from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BinaryauthorizationProjectsPlatformsPoliciesCreateRequest(_messages.Message):
    """A BinaryauthorizationProjectsPlatformsPoliciesCreateRequest object.

  Fields:
    parent: Required. The parent of this platform policy.
    platformPolicy: A PlatformPolicy resource to be passed as the request
      body.
    policyId: Required. The platform policy ID.
  """
    parent = _messages.StringField(1, required=True)
    platformPolicy = _messages.MessageField('PlatformPolicy', 2)
    policyId = _messages.StringField(3)