from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoverySchemaModifiedCadence(_messages.Message):
    """The cadence at which to update data profiles when a schema is modified.

  Enums:
    FrequencyValueValuesEnum: How frequently profiles may be updated when
      schemas are modified. Defaults to monthly.
    TypesValueListEntryValuesEnum:

  Fields:
    frequency: How frequently profiles may be updated when schemas are
      modified. Defaults to monthly.
    types: The type of events to consider when deciding if the table's schema
      has been modified and should have the profile updated. Defaults to
      NEW_COLUMNS.
  """

    class FrequencyValueValuesEnum(_messages.Enum):
        """How frequently profiles may be updated when schemas are modified.
    Defaults to monthly.

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
      SCHEMA_MODIFICATION_UNSPECIFIED: Unused
      SCHEMA_NEW_COLUMNS: Profiles should be regenerated when new columns are
        added to the table. Default.
      SCHEMA_REMOVED_COLUMNS: Profiles should be regenerated when columns are
        removed from the table.
    """
        SCHEMA_MODIFICATION_UNSPECIFIED = 0
        SCHEMA_NEW_COLUMNS = 1
        SCHEMA_REMOVED_COLUMNS = 2
    frequency = _messages.EnumField('FrequencyValueValuesEnum', 1)
    types = _messages.EnumField('TypesValueListEntryValuesEnum', 2, repeated=True)