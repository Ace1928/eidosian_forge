from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoveryCloudSqlGenerationCadence(_messages.Message):
    """How often existing tables should have their profiles refreshed. New
  tables are scanned as quickly as possible depending on system capacity.

  Enums:
    RefreshFrequencyValueValuesEnum: Data changes (non-schema changes) in
      Cloud SQL tables can't trigger reprofiling. If you set this field,
      profiles are refreshed at this frequency regardless of whether the
      underlying tables have changes. Defaults to never.

  Fields:
    refreshFrequency: Data changes (non-schema changes) in Cloud SQL tables
      can't trigger reprofiling. If you set this field, profiles are refreshed
      at this frequency regardless of whether the underlying tables have
      changes. Defaults to never.
    schemaModifiedCadence: When to reprofile if the schema has changed.
  """

    class RefreshFrequencyValueValuesEnum(_messages.Enum):
        """Data changes (non-schema changes) in Cloud SQL tables can't trigger
    reprofiling. If you set this field, profiles are refreshed at this
    frequency regardless of whether the underlying tables have changes.
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
    refreshFrequency = _messages.EnumField('RefreshFrequencyValueValuesEnum', 1)
    schemaModifiedCadence = _messages.MessageField('GooglePrivacyDlpV2SchemaModifiedCadence', 2)