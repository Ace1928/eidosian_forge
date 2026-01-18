from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShardByValueValuesEnum(_messages.Enum):
    """Mechanism used to determine which version a request is sent to. The
    traffic selection algorithm will be stable for either type until
    allocations are changed.

    Values:
      UNSPECIFIED: Diversion method unspecified.
      COOKIE: Diversion based on a specially named cookie, "GOOGAPPUID." The
        cookie must be set by the application itself or no diversion will
        occur.
      IP: Diversion based on applying the modulus operation to a fingerprint
        of the IP address.
      RANDOM: Diversion based on weighted random assignment. An incoming
        request is randomly routed to a version in the traffic split, with
        probability proportional to the version's traffic share.
    """
    UNSPECIFIED = 0
    COOKIE = 1
    IP = 2
    RANDOM = 3