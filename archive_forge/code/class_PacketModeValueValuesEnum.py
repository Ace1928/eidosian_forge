from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PacketModeValueValuesEnum(_messages.Enum):
    """The BFD packet mode for this BGP peer. If set to CONTROL_AND_ECHO, BFD
    echo mode is enabled for this BGP peer. In this mode, if the peer router
    also has BFD echo mode enabled, BFD echo packets will be sent to the other
    router. If the peer router does not have BFD echo mode enabled, only
    control packets will be sent. If set to CONTROL_ONLY, BFD echo mode is
    disabled for this BGP peer. If this router and the peer router have a
    multihop connection, this should be set to CONTROL_ONLY as BFD echo mode
    is only supported on singlehop connections. The default is
    CONTROL_AND_ECHO.

    Values:
      CONTROL_AND_ECHO: <no description>
      CONTROL_ONLY: <no description>
    """
    CONTROL_AND_ECHO = 0
    CONTROL_ONLY = 1