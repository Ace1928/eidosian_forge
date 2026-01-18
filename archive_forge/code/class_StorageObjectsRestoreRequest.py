from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageObjectsRestoreRequest(_messages.Message):
    """A StorageObjectsRestoreRequest object.

  Enums:
    ProjectionValueValuesEnum: Set of properties to return. Defaults to full.

  Fields:
    bucket: Name of the bucket in which the object resides.
    copySourceAcl: If true, copies the source object's ACL; otherwise, uses
      the bucket's default object ACL. The default is false.
    generation: Selects a specific revision of this object.
    ifGenerationMatch: Makes the operation conditional on whether the object's
      one live generation matches the given value. Setting to 0 makes the
      operation succeed only if there are no live versions of the object.
    ifGenerationNotMatch: Makes the operation conditional on whether none of
      the object's live generations match the given value. If no live object
      exists, the precondition fails. Setting to 0 makes the operation succeed
      only if there is a live version of the object.
    ifMetagenerationMatch: Makes the operation conditional on whether the
      object's one live metageneration matches the given value.
    ifMetagenerationNotMatch: Makes the operation conditional on whether none
      of the object's live metagenerations match the given value.
    object: Name of the object. For information about how to URL encode object
      names to be path safe, see Encoding URI Path Parts.
    projection: Set of properties to return. Defaults to full.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """

    class ProjectionValueValuesEnum(_messages.Enum):
        """Set of properties to return. Defaults to full.

    Values:
      full: Include all properties.
      noAcl: Omit the owner, acl property.
    """
        full = 0
        noAcl = 1
    bucket = _messages.StringField(1, required=True)
    copySourceAcl = _messages.BooleanField(2)
    generation = _messages.IntegerField(3, required=True)
    ifGenerationMatch = _messages.IntegerField(4)
    ifGenerationNotMatch = _messages.IntegerField(5)
    ifMetagenerationMatch = _messages.IntegerField(6)
    ifMetagenerationNotMatch = _messages.IntegerField(7)
    object = _messages.StringField(8, required=True)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 9)
    userProject = _messages.StringField(10)