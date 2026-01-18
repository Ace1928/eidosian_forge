from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlterMetadataResourceLocationRequest(_messages.Message):
    """Request message for DataprocMetastore.AlterMetadataResourceLocation.

  Fields:
    locationUri: Required. The new location URI for the metadata resource.
    resourceName: Required. The relative metadata resource name in the
      following format.databases/{database_id} or
      databases/{database_id}/tables/{table_id} or
      databases/{database_id}/tables/{table_id}/partitions/{partition_id}
  """
    locationUri = _messages.StringField(1)
    resourceName = _messages.StringField(2)