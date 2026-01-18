from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageObjectsCopyRequest(_messages.Message):
    """A StorageObjectsCopyRequest object.

  Enums:
    DestinationPredefinedAclValueValuesEnum: Apply a predefined set of access
      controls to the destination object.
    ProjectionValueValuesEnum: Set of properties to return. Defaults to noAcl,
      unless the object resource specifies the acl property, when it defaults
      to full.

  Fields:
    destinationBucket: Name of the bucket in which to store the new object.
      Overrides the provided object metadata's bucket value, if any.For
      information about how to URL encode object names to be path safe, see
      Encoding URI Path Parts.
    destinationObject: Name of the new object. Required when the object
      metadata is not otherwise provided. Overrides the object metadata's name
      value, if any.
    destinationPredefinedAcl: Apply a predefined set of access controls to the
      destination object.
    ifGenerationMatch: Makes the operation conditional on whether the
      destination object's current generation matches the given value. Setting
      to 0 makes the operation succeed only if there are no live versions of
      the object.
    ifGenerationNotMatch: Makes the operation conditional on whether the
      destination object's current generation does not match the given value.
      If no live object exists, the precondition fails. Setting to 0 makes the
      operation succeed only if there is a live version of the object.
    ifMetagenerationMatch: Makes the operation conditional on whether the
      destination object's current metageneration matches the given value.
    ifMetagenerationNotMatch: Makes the operation conditional on whether the
      destination object's current metageneration does not match the given
      value.
    ifSourceGenerationMatch: Makes the operation conditional on whether the
      source object's current generation matches the given value.
    ifSourceGenerationNotMatch: Makes the operation conditional on whether the
      source object's current generation does not match the given value.
    ifSourceMetagenerationMatch: Makes the operation conditional on whether
      the source object's current metageneration matches the given value.
    ifSourceMetagenerationNotMatch: Makes the operation conditional on whether
      the source object's current metageneration does not match the given
      value.
    object: A Object resource to be passed as the request body.
    projection: Set of properties to return. Defaults to noAcl, unless the
      object resource specifies the acl property, when it defaults to full.
    sourceBucket: Name of the bucket in which to find the source object.
    sourceGeneration: If present, selects a specific revision of the source
      object (as opposed to the latest version, the default).
    sourceObject: Name of the source object. For information about how to URL
      encode object names to be path safe, see Encoding URI Path Parts.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """

    class DestinationPredefinedAclValueValuesEnum(_messages.Enum):
        """Apply a predefined set of access controls to the destination object.

    Values:
      authenticatedRead: Object owner gets OWNER access, and
        allAuthenticatedUsers get READER access.
      bucketOwnerFullControl: Object owner gets OWNER access, and project team
        owners get OWNER access.
      bucketOwnerRead: Object owner gets OWNER access, and project team owners
        get READER access.
      private: Object owner gets OWNER access.
      projectPrivate: Object owner gets OWNER access, and project team members
        get access according to their roles.
      publicRead: Object owner gets OWNER access, and allUsers get READER
        access.
    """
        authenticatedRead = 0
        bucketOwnerFullControl = 1
        bucketOwnerRead = 2
        private = 3
        projectPrivate = 4
        publicRead = 5

    class ProjectionValueValuesEnum(_messages.Enum):
        """Set of properties to return. Defaults to noAcl, unless the object
    resource specifies the acl property, when it defaults to full.

    Values:
      full: Include all properties.
      noAcl: Omit the owner, acl property.
    """
        full = 0
        noAcl = 1
    destinationBucket = _messages.StringField(1, required=True)
    destinationObject = _messages.StringField(2, required=True)
    destinationPredefinedAcl = _messages.EnumField('DestinationPredefinedAclValueValuesEnum', 3)
    ifGenerationMatch = _messages.IntegerField(4)
    ifGenerationNotMatch = _messages.IntegerField(5)
    ifMetagenerationMatch = _messages.IntegerField(6)
    ifMetagenerationNotMatch = _messages.IntegerField(7)
    ifSourceGenerationMatch = _messages.IntegerField(8)
    ifSourceGenerationNotMatch = _messages.IntegerField(9)
    ifSourceMetagenerationMatch = _messages.IntegerField(10)
    ifSourceMetagenerationNotMatch = _messages.IntegerField(11)
    object = _messages.MessageField('Object', 12)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 13)
    sourceBucket = _messages.StringField(14, required=True)
    sourceGeneration = _messages.IntegerField(15)
    sourceObject = _messages.StringField(16, required=True)
    userProject = _messages.StringField(17)