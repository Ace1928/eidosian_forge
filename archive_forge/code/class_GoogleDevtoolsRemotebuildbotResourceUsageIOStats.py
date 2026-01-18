from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildbotResourceUsageIOStats(_messages.Message):
    """A GoogleDevtoolsRemotebuildbotResourceUsageIOStats object.

  Fields:
    readBytesCount: A string attribute.
    readCount: A string attribute.
    readTimeMs: A string attribute.
    writeBytesCount: A string attribute.
    writeCount: A string attribute.
    writeTimeMs: A string attribute.
  """
    readBytesCount = _messages.IntegerField(1, variant=_messages.Variant.UINT64)
    readCount = _messages.IntegerField(2, variant=_messages.Variant.UINT64)
    readTimeMs = _messages.IntegerField(3, variant=_messages.Variant.UINT64)
    writeBytesCount = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    writeCount = _messages.IntegerField(5, variant=_messages.Variant.UINT64)
    writeTimeMs = _messages.IntegerField(6, variant=_messages.Variant.UINT64)