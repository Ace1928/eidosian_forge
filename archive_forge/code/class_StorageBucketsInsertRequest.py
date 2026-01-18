from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketsInsertRequest(_messages.Message):
    """A StorageBucketsInsertRequest object.

  Enums:
    PredefinedAclValueValuesEnum: Apply a predefined set of access controls to
      this bucket.
    PredefinedDefaultObjectAclValueValuesEnum: Apply a predefined set of
      default object access controls to this bucket.
    ProjectionValueValuesEnum: Set of properties to return. Defaults to noAcl,
      unless the bucket resource specifies acl or defaultObjectAcl properties,
      when it defaults to full.

  Fields:
    bucket: A Bucket resource to be passed as the request body.
    predefinedAcl: Apply a predefined set of access controls to this bucket.
    predefinedDefaultObjectAcl: Apply a predefined set of default object
      access controls to this bucket.
    project: A valid API project identifier.
    projection: Set of properties to return. Defaults to noAcl, unless the
      bucket resource specifies acl or defaultObjectAcl properties, when it
      defaults to full.
    userProject: The project to be billed for this request.
  """

    class PredefinedAclValueValuesEnum(_messages.Enum):
        """Apply a predefined set of access controls to this bucket.

    Values:
      authenticatedRead: Project team owners get OWNER access, and
        allAuthenticatedUsers get READER access.
      private: Project team owners get OWNER access.
      projectPrivate: Project team members get access according to their
        roles.
      publicRead: Project team owners get OWNER access, and allUsers get
        READER access.
      publicReadWrite: Project team owners get OWNER access, and allUsers get
        WRITER access.
    """
        authenticatedRead = 0
        private = 1
        projectPrivate = 2
        publicRead = 3
        publicReadWrite = 4

    class PredefinedDefaultObjectAclValueValuesEnum(_messages.Enum):
        """Apply a predefined set of default object access controls to this
    bucket.

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
        """Set of properties to return. Defaults to noAcl, unless the bucket
    resource specifies acl or defaultObjectAcl properties, when it defaults to
    full.

    Values:
      full: Include all properties.
      noAcl: Omit owner, acl and defaultObjectAcl properties.
    """
        full = 0
        noAcl = 1
    bucket = _messages.MessageField('Bucket', 1)
    predefinedAcl = _messages.EnumField('PredefinedAclValueValuesEnum', 2)
    predefinedDefaultObjectAcl = _messages.EnumField('PredefinedDefaultObjectAclValueValuesEnum', 3)
    project = _messages.StringField(4, required=True)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 5)
    userProject = _messages.StringField(6)