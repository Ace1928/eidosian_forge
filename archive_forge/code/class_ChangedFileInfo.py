from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ChangedFileInfo(_messages.Message):
    """Represents file information.

  Enums:
    OperationValueValuesEnum: The operation type for the file.

  Fields:
    fromPath: Related file path for copies or renames.  For copies, the type
      will be ADDED and the from_path will point to the source of the copy.
      For renames, the type will be ADDED, the from_path will point to the
      source of the rename, and another ChangedFileInfo record with that path
      will appear with type DELETED. In other words, a rename is represented
      as a copy plus a delete of the old path.
    hash: A hex-encoded hash for the file. Not necessarily a hash of the
      file's contents. Two paths in the same revision with the same hash have
      the same contents with high probability. Empty if the operation is
      CONFLICTED.
    operation: The operation type for the file.
    path: The path of the file.
  """

    class OperationValueValuesEnum(_messages.Enum):
        """The operation type for the file.

    Values:
      OPERATION_UNSPECIFIED: No operation was specified.
      ADDED: The file was added.
      DELETED: The file was deleted.
      MODIFIED: The file was modified.
      CONFLICTED: The result of merging the file is a conflict. The CONFLICTED
        type only appears in Workspace.changed_files or Snapshot.changed_files
        when the workspace is in a merge state.
    """
        OPERATION_UNSPECIFIED = 0
        ADDED = 1
        DELETED = 2
        MODIFIED = 3
        CONFLICTED = 4
    fromPath = _messages.StringField(1)
    hash = _messages.StringField(2)
    operation = _messages.EnumField('OperationValueValuesEnum', 3)
    path = _messages.StringField(4)