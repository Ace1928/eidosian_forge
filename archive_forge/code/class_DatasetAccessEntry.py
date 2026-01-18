from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatasetAccessEntry(_messages.Message):
    """Grants all resources of particular types in a particular dataset read
  access to the current dataset. Similar to how individually authorized views
  work, updates to any resource granted through its dataset (including
  creation of new resources) requires read permission to referenced resources,
  plus write permission to the authorizing dataset.

  Enums:
    TargetTypesValueListEntryValuesEnum:

  Fields:
    dataset: The dataset this entry applies to
    targetTypes: Which resources in the dataset this entry applies to.
      Currently, only views are supported, but additional target types may be
      added in the future.
  """

    class TargetTypesValueListEntryValuesEnum(_messages.Enum):
        """TargetTypesValueListEntryValuesEnum enum type.

    Values:
      TARGET_TYPE_UNSPECIFIED: Do not use. You must set a target type
        explicitly.
      VIEWS: This entry applies to views in the dataset.
      ROUTINES: This entry applies to routines in the dataset.
    """
        TARGET_TYPE_UNSPECIFIED = 0
        VIEWS = 1
        ROUTINES = 2
    dataset = _messages.MessageField('DatasetReference', 1)
    targetTypes = _messages.EnumField('TargetTypesValueListEntryValuesEnum', 2, repeated=True)