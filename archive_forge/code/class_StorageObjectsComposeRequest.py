from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageObjectsComposeRequest(_messages.Message):
    """A StorageObjectsComposeRequest object.

  Enums:
    DestinationPredefinedAclValueValuesEnum: Apply a predefined set of access
      controls to the destination object.

  Fields:
    composeRequest: A ComposeRequest resource to be passed as the request
      body.
    destinationBucket: Name of the bucket containing the source objects. The
      destination object is stored in this bucket.
    destinationObject: Name of the new object. For information about how to
      URL encode object names to be path safe, see Encoding URI Path Parts.
    destinationPredefinedAcl: Apply a predefined set of access controls to the
      destination object.
    ifGenerationMatch: Makes the operation conditional on whether the object's
      current generation matches the given value. Setting to 0 makes the
      operation succeed only if there are no live versions of the object.
    ifMetagenerationMatch: Makes the operation conditional on whether the
      object's current metageneration matches the given value.
    kmsKeyName: Resource name of the Cloud KMS key, of the form projects/my-
      project/locations/global/keyRings/my-kr/cryptoKeys/my-key, that will be
      used to encrypt the object. Overrides the object metadata's kms_key_name
      value, if any.
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
    composeRequest = _messages.MessageField('ComposeRequest', 1)
    destinationBucket = _messages.StringField(2, required=True)
    destinationObject = _messages.StringField(3, required=True)
    destinationPredefinedAcl = _messages.EnumField('DestinationPredefinedAclValueValuesEnum', 4)
    ifGenerationMatch = _messages.IntegerField(5)
    ifMetagenerationMatch = _messages.IntegerField(6)
    kmsKeyName = _messages.StringField(7)
    userProject = _messages.StringField(8)