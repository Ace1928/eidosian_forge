from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p3beta1Property(_messages.Message):
    """A `Property` consists of a user-supplied name/value pair.

  Fields:
    name: Name of the property.
    uint64Value: Value of numeric properties.
    value: Value of the property.
  """
    name = _messages.StringField(1)
    uint64Value = _messages.IntegerField(2, variant=_messages.Variant.UINT64)
    value = _messages.StringField(3)