from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PowerIPAddressRanges(_messages.Message):
    """A PowerIPAddress Range

  Fields:
    endingIpAddress: The ending IP address of the network in IPv4 format.
    startingIpAddress: The starting IP address of the network in IPv4 format.
  """
    endingIpAddress = _messages.StringField(1)
    startingIpAddress = _messages.StringField(2)