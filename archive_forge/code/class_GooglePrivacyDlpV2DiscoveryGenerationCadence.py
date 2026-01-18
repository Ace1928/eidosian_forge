from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoveryGenerationCadence(_messages.Message):
    """What must take place for a profile to be updated and how frequently it
  should occur. New tables are scanned as quickly as possible depending on
  system capacity.

  Fields:
    schemaModifiedCadence: Governs when to update data profiles when a schema
      is modified.
    tableModifiedCadence: Governs when to update data profiles when a table is
      modified.
  """
    schemaModifiedCadence = _messages.MessageField('GooglePrivacyDlpV2DiscoverySchemaModifiedCadence', 1)
    tableModifiedCadence = _messages.MessageField('GooglePrivacyDlpV2DiscoveryTableModifiedCadence', 2)