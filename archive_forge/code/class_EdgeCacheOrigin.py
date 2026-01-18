from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgeCacheOrigin(_messages.Message):
    """Represents an HTTP-reachable backend for an `EdgeCacheService` resource.

  Enums:
    ProtocolValueValuesEnum: Optional. The protocol to use to connect to the
      configured origin. Defaults to HTTP2, which is strongly recommended for
      both security and performance. When using HTTP2 or HTTPS as the
      protocol, a valid, publicly-signed, unexpired TLS (SSL) certificate must
      be presented by the origin server.
    RetryConditionsValueListEntryValuesEnum:

  Messages:
    LabelsValue: Optional. A set of label tags associated with the
      EdgeCacheOrigin resource.

  Fields:
    awsV4Authentication: Optional. Enable AWS Signature Version 4 origin
      authentication.
    createTime: Output only. The creation timestamp in RFC3339 text format.
    description: Optional. A human-readable description of the resource.
    failoverOrigin: Optional. The EdgeCacheOrigin resource to try when the
      current origin cannot be reached. After max_attempts is reached, the
      configured failover_origin is used to fulfil the request. The following
      are both valid paths to an `EdgeCacheOrigin` resource: * `projects/my-
      project/locations/global/edgeCacheOrigins/my-origin` * `my-origin` The
      value of max_attempts_timeout dictates the timeout across all origins.
    labels: Optional. A set of label tags associated with the EdgeCacheOrigin
      resource.
    maxAttempts: Optional. The maximum number of attempts to cache fill from
      this origin. Another attempt is made when a cache fill fails with one of
      the retry_conditions or following a redirect response matching one of
      the origin_redirect.redirect_conditions. Once the maximum attempts to
      this origin have failed, the failover origin][], if specified, is used.
      The failover origin can have its own `max_attempts`, `retry_conditions`,
      `redirect_conditions`, and `failover_origin` values to control its cache
      fill failures. The total number of allowed attempts to cache fill across
      this and failover origins is limited to four. The total time allowed for
      cache fill attempts across this and failover origins can be controlled
      with `max_attempts_timeout`. The last valid, non-retried response from
      all origins is returned to the client. If no origin returns a valid
      response, an `HTTP 502` error is returned to the client. Defaults to 1.
      Must be a value greater than 0 and less than 5.
    name: Required. The name of the resource as provided by the client when
      the resource is created. The name must be 1-64 characters long, and
      match the regular expression `[a-zA-Z]([a-zA-Z0-9_-])*`, which means
      that the first character must be a letter, and all following characters
      must be a dash, an underscore, a letter, or a digit.
    originAddress: Required. A fully qualified domain name (FQDN), an IPv4 or
      IPv6 address reachable over the public internet, or the address of a
      Google Cloud Storage bucket. This address is used as the origin for
      cache requests. The following are example origins: - **FQDN**: `media-
      backend.example.com` - **IPv4**: `35.218.1.1` - **IPv6**:
      `2607:f8b0:4012:809::200e` - **Google Cloud Storage**: `gs://bucketname`
      or `bucketname.storage.googleapis.com` The following limitations apply
      to fully-qualified domain names: * They must be resolvable through
      public DNS. * They must not contain a protocol (such as `https://`). *
      They must not contain any slashes. When providing an IP address, it must
      be publicly routable. IPv6 addresses must not be enclosed in square
      brackets.
    originOverrideAction: Optional. The override actions, including URL
      rewrites and header additions, for requests that use this origin.
    originRedirect: Optional. Follow redirects from this origin.
    port: Optional. The port to connect to the origin on. Defaults to port
      **443** for HTTP2 and HTTPS protocols and port **80** for HTTP.
    protocol: Optional. The protocol to use to connect to the configured
      origin. Defaults to HTTP2, which is strongly recommended for both
      security and performance. When using HTTP2 or HTTPS as the protocol, a
      valid, publicly-signed, unexpired TLS (SSL) certificate must be
      presented by the origin server.
    retryConditions: Optional. Specifies one or more retry conditions for the
      configured origin. If the failure mode during a connection attempt to
      the origin matches the configured `retryConditions` values, the origin
      request retries up to max_attempts times. The failover origin, if
      configured, is then used to satisfy the request. The default
      `retry_conditions` value is `CONNECT_FAILURE`. `retry_conditions` values
      apply to this origin, and not to subsequent failover origins, which can
      specify their own `retry_conditions` and `max_attempts` values. For a
      list of valid values, see RetryConditions.
    timeout: Optional. The connection and HTTP timeout configuration for this
      origin.
    updateTime: Output only. The update timestamp in RFC3339 text format.
  """

    class ProtocolValueValuesEnum(_messages.Enum):
        """Optional. The protocol to use to connect to the configured origin.
    Defaults to HTTP2, which is strongly recommended for both security and
    performance. When using HTTP2 or HTTPS as the protocol, a valid, publicly-
    signed, unexpired TLS (SSL) certificate must be presented by the origin
    server.

    Values:
      PROTOCOL_UNSPECIFIED: Unspecified value. Defaults to HTTP2.
      HTTP2: The HTTP/2 protocol. HTTP/2 refers to "h2", which requires TLS
        (HTTPS). Requires a valid (public and unexpired) TLS certificate
        present on the origin.
      HTTPS: HTTP/1.1 with TLS (SSL). Requires a valid (public and unexpired)
        TLS certificate present on the origin.
      HTTP: HTTP without TLS (SSL). This is not recommended, because
        communication outside of Google's network is unencrypted to the public
        endpoint (origin).
    """
        PROTOCOL_UNSPECIFIED = 0
        HTTP2 = 1
        HTTPS = 2
        HTTP = 3

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. A set of label tags associated with the EdgeCacheOrigin
    resource.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    awsV4Authentication = _messages.MessageField('AWSV4Signature', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    failoverOrigin = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    maxAttempts = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    name = _messages.StringField(7)
    originAddress = _messages.StringField(8)
    originOverrideAction = _messages.MessageField('OriginOverrideAction', 9)
    originRedirect = _messages.MessageField('OriginRedirect', 10)
    port = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    protocol = _messages.EnumField('ProtocolValueValuesEnum', 12)
    retryConditions = _messages.EnumField('RetryConditionsValueListEntryValuesEnum', 13, repeated=True)
    timeout = _messages.MessageField('Timeout', 14)
    updateTime = _messages.StringField(15)