from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkLLDPStatus(_messages.Message):
    """Describing a LLDP link.

  Fields:
    peerChassisId: The peer chassis component of the endpoint identifier
      associated with the transmitting LLDP agent.
    peerChassisIdType: The format and source of the peer chassis identifier
      string.
    peerPortId: The port component of the endpoint identifier associated with
      the transmitting LLDP agent. If the specified port is an IEEE 802.3
      Repeater port, then this TLV is optional.
    peerPortIdType: The format and source of the peer port identifier string.
    peerSystemDescription: The textual description of the network entity of
      LLDP peer.
    peerSystemName: The peer system's administratively assigned name.
  """
    peerChassisId = _messages.StringField(1)
    peerChassisIdType = _messages.StringField(2)
    peerPortId = _messages.StringField(3)
    peerPortIdType = _messages.StringField(4)
    peerSystemDescription = _messages.StringField(5)
    peerSystemName = _messages.StringField(6)