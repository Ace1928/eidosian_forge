from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetryConditionsValueListEntryValuesEnum(_messages.Enum):
    """RetryConditionsValueListEntryValuesEnum enum type.

    Values:
      RETRY_CONDITIONS_UNSPECIFIED: Unspecified
      CONNECT_FAILURE: Retry on failures connecting to origins include
        routing, DNS and TLS handshake errors, and TCP/UDP timeouts.
      HTTP_5XX: Retry if the origin responds with any `HTTP 5xx` response
        code.
      GATEWAY_ERROR: Similar to `5xx`, but only applies to HTTP response codes
        `502`, `503`, or `504`.
      RETRIABLE_4XX: Retry for retriable `4xx` response codes, which include
        `HTTP 409 (Conflict)` and `HTTP 429 (Too Many Requests)`.
      NOT_FOUND: Retry if the origin returns an `HTTP 404 (Not Found)` error.
        This can be useful when generating video content when the segment is
        not yet available.
      FORBIDDEN: Retry if the origin returns an `HTTP 403 (Forbidden)` error.
        This can be useful for origins that return `403` (instead of `404`)
        for missing content for security reasons.
    """
    RETRY_CONDITIONS_UNSPECIFIED = 0
    CONNECT_FAILURE = 1
    HTTP_5XX = 2
    GATEWAY_ERROR = 3
    RETRIABLE_4XX = 4
    NOT_FOUND = 5
    FORBIDDEN = 6