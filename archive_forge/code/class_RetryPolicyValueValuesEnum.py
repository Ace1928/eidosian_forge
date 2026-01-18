from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetryPolicyValueValuesEnum(_messages.Enum):
    """Optional. If unset, then defaults to ignoring failures (i.e. not
    retrying them).

    Values:
      RETRY_POLICY_UNSPECIFIED: Not specified.
      RETRY_POLICY_DO_NOT_RETRY: Do not retry.
      RETRY_POLICY_RETRY: Retry on any failure, retry up to 7 days with an
        exponential backoff (capped at 10 seconds).
    """
    RETRY_POLICY_UNSPECIFIED = 0
    RETRY_POLICY_DO_NOT_RETRY = 1
    RETRY_POLICY_RETRY = 2