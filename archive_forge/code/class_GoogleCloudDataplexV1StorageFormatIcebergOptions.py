from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1StorageFormatIcebergOptions(_messages.Message):
    """Describes Iceberg data format.

  Fields:
    metadataLocation: Optional. The location of where the iceberg metadata is
      present, must be within the table path
  """
    metadataLocation = _messages.StringField(1)