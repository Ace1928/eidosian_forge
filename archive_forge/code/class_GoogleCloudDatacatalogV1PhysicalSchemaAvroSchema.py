from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1PhysicalSchemaAvroSchema(_messages.Message):
    """Schema in Avro JSON format.

  Fields:
    text: JSON source of the Avro schema.
  """
    text = _messages.StringField(1)