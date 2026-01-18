from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocalStateValueValuesEnum(_messages.Enum):
    """The current BFD session state as seen by the transmitting system.
    These states are specified in section 4.1 of RFC5880

    Values:
      ADMIN_DOWN: <no description>
      DOWN: <no description>
      INIT: <no description>
      STATE_UNSPECIFIED: <no description>
      UP: <no description>
    """
    ADMIN_DOWN = 0
    DOWN = 1
    INIT = 2
    STATE_UNSPECIFIED = 3
    UP = 4