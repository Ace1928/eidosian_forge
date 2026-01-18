from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectLocationRegionInfo(_messages.Message):
    """Information about any potential InterconnectAttachments between an
  Interconnect at a specific InterconnectLocation, and a specific Cloud
  Region.

  Enums:
    LocationPresenceValueValuesEnum: Identifies the network presence of this
      location.

  Fields:
    expectedRttMs: Expected round-trip time in milliseconds, from this
      InterconnectLocation to a VM in this region.
    locationPresence: Identifies the network presence of this location.
    region: URL for the region of this location.
  """

    class LocationPresenceValueValuesEnum(_messages.Enum):
        """Identifies the network presence of this location.

    Values:
      GLOBAL: This region is not in any common network presence with this
        InterconnectLocation.
      LOCAL_REGION: This region shares the same regional network presence as
        this InterconnectLocation.
      LP_GLOBAL: [Deprecated] This region is not in any common network
        presence with this InterconnectLocation.
      LP_LOCAL_REGION: [Deprecated] This region shares the same regional
        network presence as this InterconnectLocation.
    """
        GLOBAL = 0
        LOCAL_REGION = 1
        LP_GLOBAL = 2
        LP_LOCAL_REGION = 3
    expectedRttMs = _messages.IntegerField(1)
    locationPresence = _messages.EnumField('LocationPresenceValueValuesEnum', 2)
    region = _messages.StringField(3)