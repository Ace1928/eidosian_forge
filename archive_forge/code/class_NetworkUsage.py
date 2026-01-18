from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkUsage(_messages.Message):
    """Network with all used IP addresses.

  Fields:
    network: Network.
    usedIps: All used IP addresses in this network.
  """
    network = _messages.MessageField('Network', 1)
    usedIps = _messages.StringField(2, repeated=True)