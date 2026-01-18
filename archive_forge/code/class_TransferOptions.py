from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferOptions(_messages.Message):
    """TransferOptions define the actions to be performed on objects in a
  transfer.

  Enums:
    OverwriteWhenValueValuesEnum: When to overwrite objects that already exist
      in the sink. If not set, overwrite behavior is determined by
      overwrite_objects_already_existing_in_sink.

  Fields:
    deleteObjectsFromSourceAfterTransfer: Whether objects should be deleted
      from the source after they are transferred to the sink. **Note:** This
      option and delete_objects_unique_in_sink are mutually exclusive.
    deleteObjectsUniqueInSink: Whether objects that exist only in the sink
      should be deleted. **Note:** This option and
      delete_objects_from_source_after_transfer are mutually exclusive.
    metadataOptions: Represents the selected metadata options for a transfer
      job.
    overwriteObjectsAlreadyExistingInSink: When to overwrite objects that
      already exist in the sink. The default is that only objects that are
      different from the source are ovewritten. If true, all objects in the
      sink whose name matches an object in the source are overwritten with the
      source object.
    overwriteWhen: When to overwrite objects that already exist in the sink.
      If not set, overwrite behavior is determined by
      overwrite_objects_already_existing_in_sink.
  """

    class OverwriteWhenValueValuesEnum(_messages.Enum):
        """When to overwrite objects that already exist in the sink. If not set,
    overwrite behavior is determined by
    overwrite_objects_already_existing_in_sink.

    Values:
      OVERWRITE_WHEN_UNSPECIFIED: Overwrite behavior is unspecified.
      DIFFERENT: Overwrites destination objects with the source objects, only
        if the objects have the same name but different HTTP ETags or checksum
        values.
      NEVER: Never overwrites a destination object if a source object has the
        same name. In this case, the source object is not transferred.
      ALWAYS: Always overwrite the destination object with the source object,
        even if the HTTP Etags or checksum values are the same.
    """
        OVERWRITE_WHEN_UNSPECIFIED = 0
        DIFFERENT = 1
        NEVER = 2
        ALWAYS = 3
    deleteObjectsFromSourceAfterTransfer = _messages.BooleanField(1)
    deleteObjectsUniqueInSink = _messages.BooleanField(2)
    metadataOptions = _messages.MessageField('MetadataOptions', 3)
    overwriteObjectsAlreadyExistingInSink = _messages.BooleanField(4)
    overwriteWhen = _messages.EnumField('OverwriteWhenValueValuesEnum', 5)