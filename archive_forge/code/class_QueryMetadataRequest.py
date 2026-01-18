from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryMetadataRequest(_messages.Message):
    """Request message for DataprocMetastore.QueryMetadata.

  Fields:
    query: Required. A read-only SQL query to execute against the metadata
      database. The query cannot change or mutate the data.
  """
    query = _messages.StringField(1)