from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageObjectsSetIamPolicyRequest(_messages.Message):
    """A StorageObjectsSetIamPolicyRequest object.

  Fields:
    bucket: Name of the bucket in which the object resides.
    generation: If present, selects a specific revision of this object (as
      opposed to the latest version, the default).
    object: Name of the object. For information about how to URL encode object
      names to be path safe, see Encoding URI Path Parts.
    policy: A Policy resource to be passed as the request body.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    generation = _messages.IntegerField(2)
    object = _messages.StringField(3, required=True)
    policy = _messages.MessageField('Policy', 4)
    userProject = _messages.StringField(5)