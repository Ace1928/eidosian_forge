from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SetIamPolicyRequest(_messages.Message):
    """Request message for `SetIamPolicy` method.

  Fields:
    policy: REQUIRED: The complete policy to be applied to the `resource`. The
      size of the policy is limited to a few 10s of KB. An empty policy is a
      valid policy but certain Cloud Platform services (such as Projects)
      might reject them.
  """
    policy = _messages.MessageField('Policy', 1)