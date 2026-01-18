from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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