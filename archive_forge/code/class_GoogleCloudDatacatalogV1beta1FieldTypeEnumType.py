from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1FieldTypeEnumType(_messages.Message):
    """A GoogleCloudDatacatalogV1beta1FieldTypeEnumType object.

  Fields:
    allowedValues: A GoogleCloudDatacatalogV1beta1FieldTypeEnumTypeEnumValue
      attribute.
  """
    allowedValues = _messages.MessageField('GoogleCloudDatacatalogV1beta1FieldTypeEnumTypeEnumValue', 1, repeated=True)