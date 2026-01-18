from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AvroOptions(_messages.Message):
    """Options for external data sources.

  Fields:
    useAvroLogicalTypes: Optional. If sourceFormat is set to "AVRO", indicates
      whether to interpret logical types as the corresponding BigQuery data
      type (for example, TIMESTAMP), instead of using the raw type (for
      example, INTEGER).
  """
    useAvroLogicalTypes = _messages.BooleanField(1)