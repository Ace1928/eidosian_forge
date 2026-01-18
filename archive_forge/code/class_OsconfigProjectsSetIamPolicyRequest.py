from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsSetIamPolicyRequest(_messages.Message):
    """A OsconfigProjectsSetIamPolicyRequest object.

  Fields:
    resource: REQUIRED: The resource for which the policy is being specified.
      See the operation documentation for the appropriate value for this
      field.
    setIamPolicyRequest: A SetIamPolicyRequest resource to be passed as the
      request body.
  """
    resource = _messages.StringField(1, required=True)
    setIamPolicyRequest = _messages.MessageField('SetIamPolicyRequest', 2)