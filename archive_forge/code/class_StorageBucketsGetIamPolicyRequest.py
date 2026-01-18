from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketsGetIamPolicyRequest(_messages.Message):
    """A StorageBucketsGetIamPolicyRequest object.

  Fields:
    bucket: Name of a bucket.
    provisionalUserProject: The project to be billed for this request if the
      target bucket is requester-pays bucket.
    optionsRequestedPolicyVersion: The policy format version to be returned in
      the response.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    provisionalUserProject = _messages.StringField(2)
    optionsRequestedPolicyVersion = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    userProject = _messages.StringField(4)