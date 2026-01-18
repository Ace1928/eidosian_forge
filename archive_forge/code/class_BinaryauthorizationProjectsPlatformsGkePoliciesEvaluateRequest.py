from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BinaryauthorizationProjectsPlatformsGkePoliciesEvaluateRequest(_messages.Message):
    """A BinaryauthorizationProjectsPlatformsGkePoliciesEvaluateRequest object.

  Fields:
    evaluateGkePolicyRequest: A EvaluateGkePolicyRequest resource to be passed
      as the request body.
    name: Required. The name of the platform policy to evaluate in the format
      `projects/*/platforms/*/policies/*`.
  """
    evaluateGkePolicyRequest = _messages.MessageField('EvaluateGkePolicyRequest', 1)
    name = _messages.StringField(2, required=True)