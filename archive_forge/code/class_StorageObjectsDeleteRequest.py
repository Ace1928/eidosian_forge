from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageObjectsDeleteRequest(_messages.Message):
    """A StorageObjectsDeleteRequest object.

  Fields:
    bucket: Name of the bucket in which the object resides.
    generation: If present, permanently deletes a specific revision of this
      object (as opposed to the latest version, the default).
    ifGenerationMatch: Makes the operation conditional on whether the object's
      current generation matches the given value. Setting to 0 makes the
      operation succeed only if there are no live versions of the object.
    ifGenerationNotMatch: Makes the operation conditional on whether the
      object's current generation does not match the given value. If no live
      object exists, the precondition fails. Setting to 0 makes the operation
      succeed only if there is a live version of the object.
    ifMetagenerationMatch: Makes the operation conditional on whether the
      object's current metageneration matches the given value.
    ifMetagenerationNotMatch: Makes the operation conditional on whether the
      object's current metageneration does not match the given value.
    object: Name of the object. For information about how to URL encode object
      names to be path safe, see Encoding URI Path Parts.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    generation = _messages.IntegerField(2)
    ifGenerationMatch = _messages.IntegerField(3)
    ifGenerationNotMatch = _messages.IntegerField(4)
    ifMetagenerationMatch = _messages.IntegerField(5)
    ifMetagenerationNotMatch = _messages.IntegerField(6)
    object = _messages.StringField(7, required=True)
    userProject = _messages.StringField(8)