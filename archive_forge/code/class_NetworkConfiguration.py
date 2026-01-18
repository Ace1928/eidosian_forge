from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class NetworkConfiguration(_messages.Message):
    """A NetworkConfiguration object.

  Fields:
    downRule: The emulation rule applying to the download traffic.
    id: The unique opaque id for this network traffic configuration.
    upRule: The emulation rule applying to the upload traffic.
  """
    downRule = _messages.MessageField('TrafficRule', 1)
    id = _messages.StringField(2)
    upRule = _messages.MessageField('TrafficRule', 3)