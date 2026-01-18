from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BundledLocalSsds(_messages.Message):
    """A BundledLocalSsds object.

  Fields:
    defaultInterface: The default disk interface if the interface is not
      specified.
    partitionCount: The number of partitions.
  """
    defaultInterface = _messages.StringField(1)
    partitionCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)