from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FolderOperationError(_messages.Message):
    """A classification of the Folder Operation error.

  Enums:
    ErrorMessageIdValueValuesEnum: The type of operation error experienced.

  Fields:
    errorMessageId: The type of operation error experienced.
  """

    class ErrorMessageIdValueValuesEnum(_messages.Enum):
        """The type of operation error experienced.

    Values:
      ERROR_TYPE_UNSPECIFIED: The error type was unrecognized or unspecified.
      ACTIVE_FOLDER_HEIGHT_VIOLATION: The attempted action would violate the
        max folder depth constraint.
      MAX_CHILD_FOLDERS_VIOLATION: The attempted action would violate the max
        child folders constraint.
      FOLDER_NAME_UNIQUENESS_VIOLATION: The attempted action would violate the
        locally-unique folder display_name constraint.
      RESOURCE_DELETED_VIOLATION: The resource being moved has been deleted.
      PARENT_DELETED_VIOLATION: The resource a folder was being added to has
        been deleted.
      CYCLE_INTRODUCED_VIOLATION: The attempted action would introduce cycle
        in resource path.
      FOLDER_BEING_MOVED_VIOLATION: The attempted action would move a folder
        that is already being moved.
      FOLDER_TO_DELETE_NON_EMPTY_VIOLATION: The folder the caller is trying to
        delete contains active resources.
      DELETED_FOLDER_HEIGHT_VIOLATION: The attempted action would violate the
        max deleted folder depth constraint.
    """
        ERROR_TYPE_UNSPECIFIED = 0
        ACTIVE_FOLDER_HEIGHT_VIOLATION = 1
        MAX_CHILD_FOLDERS_VIOLATION = 2
        FOLDER_NAME_UNIQUENESS_VIOLATION = 3
        RESOURCE_DELETED_VIOLATION = 4
        PARENT_DELETED_VIOLATION = 5
        CYCLE_INTRODUCED_VIOLATION = 6
        FOLDER_BEING_MOVED_VIOLATION = 7
        FOLDER_TO_DELETE_NON_EMPTY_VIOLATION = 8
        DELETED_FOLDER_HEIGHT_VIOLATION = 9
    errorMessageId = _messages.EnumField('ErrorMessageIdValueValuesEnum', 1)