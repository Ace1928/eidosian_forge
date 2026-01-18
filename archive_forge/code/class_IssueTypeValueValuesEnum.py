from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IssueTypeValueValuesEnum(_messages.Enum):
    """Form this outage is expected to take, which can take one of the
    following values: - OUTAGE: The Interconnect may be completely out of
    service for some or all of the specified window. - PARTIAL_OUTAGE: Some
    circuits comprising the Interconnect as a whole should remain up, but with
    reduced bandwidth. Note that the versions of this enum prefixed with "IT_"
    have been deprecated in favor of the unprefixed values.

    Values:
      IT_OUTAGE: [Deprecated] The Interconnect may be completely out of
        service for some or all of the specified window.
      IT_PARTIAL_OUTAGE: [Deprecated] Some circuits comprising the
        Interconnect will be out of service during the expected window. The
        interconnect as a whole should remain up, albeit with reduced
        bandwidth.
      OUTAGE: The Interconnect may be completely out of service for some or
        all of the specified window.
      PARTIAL_OUTAGE: Some circuits comprising the Interconnect will be out of
        service during the expected window. The interconnect as a whole should
        remain up, albeit with reduced bandwidth.
    """
    IT_OUTAGE = 0
    IT_PARTIAL_OUTAGE = 1
    OUTAGE = 2
    PARTIAL_OUTAGE = 3