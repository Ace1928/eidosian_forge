from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeMachineImagesSetIamPolicyRequest(_messages.Message):
    """A ComputeMachineImagesSetIamPolicyRequest object.

  Fields:
    globalSetPolicyRequest: A GlobalSetPolicyRequest resource to be passed as
      the request body.
    project: Project ID for this request.
    resource: Name or id of the resource for this request.
  """
    globalSetPolicyRequest = _messages.MessageField('GlobalSetPolicyRequest', 1)
    project = _messages.StringField(2, required=True)
    resource = _messages.StringField(3, required=True)