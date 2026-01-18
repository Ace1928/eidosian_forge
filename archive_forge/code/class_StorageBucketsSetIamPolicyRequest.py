from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketsSetIamPolicyRequest(_messages.Message):
    """A StorageBucketsSetIamPolicyRequest object.

  Fields:
    bucket: Name of a bucket.
    policy: A Policy resource to be passed as the request body.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    policy = _messages.MessageField('Policy', 2)
    userProject = _messages.StringField(3)