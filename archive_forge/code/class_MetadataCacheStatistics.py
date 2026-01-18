from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataCacheStatistics(_messages.Message):
    """Statistics for metadata caching in BigLake tables.

  Fields:
    tableMetadataCacheUsage: Set for the Metadata caching eligible tables
      referenced in the query.
  """
    tableMetadataCacheUsage = _messages.MessageField('TableMetadataCacheUsage', 1, repeated=True)