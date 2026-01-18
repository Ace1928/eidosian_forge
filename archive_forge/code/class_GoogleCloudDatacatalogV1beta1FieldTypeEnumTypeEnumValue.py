from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1FieldTypeEnumTypeEnumValue(_messages.Message):
    """A GoogleCloudDatacatalogV1beta1FieldTypeEnumTypeEnumValue object.

  Fields:
    displayName: Required. The display name of the enum value. Must not be an
      empty string.
  """
    displayName = _messages.StringField(1)