from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoveryTableModifiedCadence(_messages.Message):
    """The cadence at which to update data profiles when a table is modified.

  Enums:
    FrequencyValueValuesEnum: How frequently data profiles can be updated when
      tables are modified. Defaults to never.
    TypesValueListEntryValuesEnum:

  Fields:
    frequency: How frequently data profiles can be updated when tables are
      modified. Defaults to never.
    types: The type of events to consider when deciding if the table has been
      modified and should have the profile updated. Defaults to
      MODIFIED_TIMESTAMP.
  """

    class FrequencyValueValuesEnum(_messages.Enum):
        """How frequently data profiles can be updated when tables are modified.
    Defaults to never.

    Values:
      UPDATE_FREQUENCY_UNSPECIFIED: Unspecified.
      UPDATE_FREQUENCY_NEVER: After the data profile is created, it will never
        be updated.
      UPDATE_FREQUENCY_DAILY: The data profile can be updated up to once every
        24 hours.
      UPDATE_FREQUENCY_MONTHLY: The data profile can be updated up to once
        every 30 days. Default.
    """
        UPDATE_FREQUENCY_UNSPECIFIED = 0
        UPDATE_FREQUENCY_NEVER = 1
        UPDATE_FREQUENCY_DAILY = 2
        UPDATE_FREQUENCY_MONTHLY = 3

    class TypesValueListEntryValuesEnum(_messages.Enum):
        """TypesValueListEntryValuesEnum enum type.

    Values:
      TABLE_MODIFICATION_UNSPECIFIED: Unused.
      TABLE_MODIFIED_TIMESTAMP: A table will be considered modified when the
        last_modified_time from BigQuery has been updated.
    """
        TABLE_MODIFICATION_UNSPECIFIED = 0
        TABLE_MODIFIED_TIMESTAMP = 1
    frequency = _messages.EnumField('FrequencyValueValuesEnum', 1)
    types = _messages.EnumField('TypesValueListEntryValuesEnum', 2, repeated=True)