from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudResourcesettingsV1ValueEnumValue(_messages.Message):
    """A enum value that can hold any enum type setting values. Each enum type
  is represented by a number, this representation is stored in the
  definitions.

  Fields:
    value: The value of this enum
  """
    value = _messages.StringField(1)