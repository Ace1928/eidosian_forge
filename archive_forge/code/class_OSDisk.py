from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSDisk(_messages.Message):
    """A message describing the OS disk.

  Fields:
    name: The disk's full name.
    sizeGb: The disk's size in GB.
    type: The disk's type.
  """
    name = _messages.StringField(1)
    sizeGb = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    type = _messages.StringField(3)