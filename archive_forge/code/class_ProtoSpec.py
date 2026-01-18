from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProtoSpec(_messages.Message):
    """A collection of protocol buffer service specification files.

  Fields:
    protoDescriptor: A complete descriptor of a protocol buffer specification
  """
    protoDescriptor = _messages.MessageField('ProtoDescriptor', 1)