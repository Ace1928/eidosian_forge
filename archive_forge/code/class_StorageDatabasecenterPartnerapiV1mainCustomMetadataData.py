from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterPartnerapiV1mainCustomMetadataData(_messages.Message):
    """Any custom metadata associated with the resource. i.e. A spanner
  instance can have multiple databases with its own unique metadata.
  Information for these individual databases can be captured in custom
  metadata data

  Fields:
    databaseMetadata: A StorageDatabasecenterPartnerapiV1mainDatabaseMetadata
      attribute.
  """
    databaseMetadata = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainDatabaseMetadata', 1, repeated=True)