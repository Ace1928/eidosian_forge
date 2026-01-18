from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BfdSessionInitializationModeValueValuesEnum(_messages.Enum):
    """The BFD session initialization mode for this BGP peer. If set to
    ACTIVE, the Cloud Router will initiate the BFD session for this BGP peer.
    If set to PASSIVE, the Cloud Router will wait for the peer router to
    initiate the BFD session for this BGP peer. If set to DISABLED, BFD is
    disabled for this BGP peer.

    Values:
      ACTIVE: <no description>
      DISABLED: <no description>
      PASSIVE: <no description>
    """
    ACTIVE = 0
    DISABLED = 1
    PASSIVE = 2