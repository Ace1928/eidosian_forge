from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Orientation(_messages.Message):
    """Screen orientation of the device.

  Fields:
    id: The id for this orientation. Example: "portrait".
    name: A human-friendly name for this orientation. Example: "portrait".
    tags: Tags for this dimension. Example: "default".
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)
    tags = _messages.StringField(3, repeated=True)