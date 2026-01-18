from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnvoyHeadersValueValuesEnum(_messages.Enum):
    """Optional. Determines if envoy will insert internal debug headers into
    upstream requests. Other Envoy headers may still be injected. By default,
    envoy will not insert any debug headers.

    Values:
      ENVOY_HEADERS_UNSPECIFIED: Defaults to NONE.
      NONE: Suppress envoy debug headers.
      DEBUG_HEADERS: Envoy will insert default internal debug headers into
        upstream requests: x-envoy-attempt-count x-envoy-is-timeout-retry
        x-envoy-expected-rq-timeout-ms x-envoy-original-path x-envoy-upstream-
        stream-duration-ms
    """
    ENVOY_HEADERS_UNSPECIFIED = 0
    NONE = 1
    DEBUG_HEADERS = 2