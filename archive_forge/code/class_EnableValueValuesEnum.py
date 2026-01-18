from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnableValueValuesEnum(_messages.Enum):
    """The status of the BGP peer connection. If set to FALSE, any active
    session with the peer is terminated and all associated routing information
    is removed. If set to TRUE, the peer connection can be established with
    routing information. The default is TRUE.

    Values:
      FALSE: <no description>
      TRUE: <no description>
    """
    FALSE = 0
    TRUE = 1