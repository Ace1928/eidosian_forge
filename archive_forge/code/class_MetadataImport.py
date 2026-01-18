from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataImport(_messages.Message):
    """A metastore resource that imports metadata.

  Enums:
    StateValueValuesEnum: Output only. The current state of the metadata
      import.

  Fields:
    createTime: Output only. The time when the metadata import was started.
    databaseDump: Immutable. A database dump from a pre-existing metastore's
      database.
    description: The description of the metadata import.
    endTime: Output only. The time when the metadata import finished.
    name: Immutable. The relative resource name of the metadata import, of the
      form:projects/{project_number}/locations/{location_id}/services/{service
      _id}/metadataImports/{metadata_import_id}.
    state: Output only. The current state of the metadata import.
    updateTime: Output only. The time when the metadata import was last
      updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the metadata import.

    Values:
      STATE_UNSPECIFIED: The state of the metadata import is unknown.
      RUNNING: The metadata import is running.
      SUCCEEDED: The metadata import completed successfully.
      UPDATING: The metadata import is being updated.
      FAILED: The metadata import failed, and attempted metadata changes were
        rolled back.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        SUCCEEDED = 2
        UPDATING = 3
        FAILED = 4
    createTime = _messages.StringField(1)
    databaseDump = _messages.MessageField('DatabaseDump', 2)
    description = _messages.StringField(3)
    endTime = _messages.StringField(4)
    name = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    updateTime = _messages.StringField(7)