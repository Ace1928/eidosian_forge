from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailureReasonValueValuesEnum(_messages.Enum):
    """Output only. Reason for failure of the authorization attempt for the
    domain.

    Values:
      FAILURE_REASON_UNSPECIFIED: <no description>
      CONFIG: There was a problem with the user's DNS or load balancer
        configuration for this domain.
      CAA: Certificate issuance forbidden by an explicit CAA record for the
        domain or a failure to check CAA records for the domain.
      RATE_LIMITED: Reached a CA or internal rate-limit for the domain, e.g.
        for certificates per top-level private domain.
    """
    FAILURE_REASON_UNSPECIFIED = 0
    CONFIG = 1
    CAA = 2
    RATE_LIMITED = 3