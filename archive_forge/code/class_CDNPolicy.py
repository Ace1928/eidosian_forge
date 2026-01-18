from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CDNPolicy(_messages.Message):
    """The CDN policy to apply to the configured route.

  Enums:
    CacheModeValueValuesEnum: Optional. Set the CacheMode used by this route.
      BYPASS_CACHE and USE_ORIGIN_HEADERS proxy the origin's headers. Other
      cache modes pass Cache-Control to the client. Use client_ttl to override
      what is sent to the client.
    SignedRequestModeValueValuesEnum: Optional. Specifies whether to enforce
      signed requests. The default value is DISABLED, which means all content
      is public, and does not authorize access. You must also set a
      signed_request_keyset to enable signed requests. When set to
      REQUIRE_SIGNATURES or REQUIRE_TOKENS, all matching requests get their
      signature validated. Requests that aren't signed with the corresponding
      private key, or that are otherwise invalid (such as expired or do not
      match the signature, IP address, or header) are rejected with an HTTP
      403 error. If logging is turned on, then invalid requests are also
      logged.

  Messages:
    NegativeCachingPolicyValue: Optional. A cache TTL for the specified HTTP
      status code. negative_caching must be enabled to configure
      `negative_caching_policy`. The following limitations apply: - Omitting
      the policy and leaving `negative_caching` enabled uses the default TTLs
      for each status code, defined in `negative_caching`. - TTLs must be >=
      `0` (where `0` is "always revalidate") and <= `86400s` (1 day) You can
      set only the following status codes: - HTTP redirection (`300`, `301`,
      `302`, `307`, or `308`) - Client error (`400`, `403`, `404`, `405`,
      `410`, `421`, or `451`) - Server error (`500`, `501`, `502`, `503`, or
      `504`) When you specify an explicit `negative_caching_policy`, ensure
      that you also specify a cache TTL for all response codes that you wish
      to cache. The CDNPolicy doesn't apply any default negative caching when
      a policy exists.

  Fields:
    addSignatures: Optional. Enables signature generation or propagation on
      this route. This field can only be specified when signed_request_mode is
      set to REQUIRE_TOKENS.
    cacheKeyPolicy: Optional. The request parameters that contribute to the
      cache key.
    cacheMode: Optional. Set the CacheMode used by this route. BYPASS_CACHE
      and USE_ORIGIN_HEADERS proxy the origin's headers. Other cache modes
      pass Cache-Control to the client. Use client_ttl to override what is
      sent to the client.
    clientTtl: Optional. Specifies a separate client (such as browser client)
      TTL, separate from the TTL used by the edge caches. Leaving this empty
      uses the same cache TTL for both the CDN and the client-facing response.
      - The TTL must be > `0` and <= `86400s` (1 day) - The `client_ttl`
      cannot be larger than the default_ttl (if set) - Fractions of a second
      are not allowed. Omit this field to use the `default_ttl`, or the max-
      age set by the origin, as the client-facing TTL. When the CacheMode is
      set to USE_ORIGIN_HEADERS or BYPASS_CACHE, you must omit this field.
    defaultTtl: Optional. Specifies the default TTL for cached content served
      by this origin for responses that do not have an existing valid TTL
      (max-age or s-max-age). Defaults to `3600s` (1 hour). - The TTL must be
      >= `0` and <= `31,536,000` seconds (1 year) - Setting a TTL of `0` means
      "always revalidate" (equivalent to must-revalidate) - The value of
      `default_ttl` cannot be set to a value greater than that of max_ttl. -
      Fractions of a second are not allowed. - When the CacheMode is set to
      FORCE_CACHE_ALL, the `default_ttl` overwrites the TTL set in all
      responses. Infrequently accessed objects might be evicted from the cache
      before the defined TTL. Objects that expire are revalidated with the
      origin. When the CacheMode is set to USE_ORIGIN_HEADERS or BYPASS_CACHE,
      you must omit this field.
    maxTtl: Optional. The maximum allowed TTL for cached content served by
      this origin. Defaults to `86400s` (1 day). Cache directives that attempt
      to set a max-age or s-maxage higher than this, or an Expires header more
      than `max_ttl` seconds in the future are capped at the value of
      `max_ttl`, as if it were the value of an s-maxage Cache-Control
      directive. - The TTL must be >= `0` and <= `31,536,000` seconds (1 year)
      - Setting a TTL of `0` means "always revalidate" - The value of
      `max_ttl` must be equal to or greater than default_ttl. - Fractions of a
      second are not allowed. When CacheMode is set to
      [USE_ORIGIN_HEADERS].[CacheMode.USE_ORIGIN_HEADERS], FORCE_CACHE_ALL, or
      BYPASS_CACHE, you must omit this field.
    negativeCaching: Optional. Negative caching allows setting per-status code
      TTLs, in order to apply fine-grained caching for common errors or
      redirects. This can reduce the load on your origin and improve end-user
      experience by reducing response latency. By default, the CDNPolicy
      applies the following default TTLs to these status codes: - **10m**:
      HTTP 300 (Multiple Choice), 301, 308 (Permanent Redirects) - **120s**:
      HTTP 404 (Not Found), 410 (Gone), 451 (Unavailable For Legal Reasons) -
      **60s**: HTTP 405 (Method Not Found), 501 (Not Implemented) These
      defaults can be overridden in negative_caching_policy
    negativeCachingPolicy: Optional. A cache TTL for the specified HTTP status
      code. negative_caching must be enabled to configure
      `negative_caching_policy`. The following limitations apply: - Omitting
      the policy and leaving `negative_caching` enabled uses the default TTLs
      for each status code, defined in `negative_caching`. - TTLs must be >=
      `0` (where `0` is "always revalidate") and <= `86400s` (1 day) You can
      set only the following status codes: - HTTP redirection (`300`, `301`,
      `302`, `307`, or `308`) - Client error (`400`, `403`, `404`, `405`,
      `410`, `421`, or `451`) - Server error (`500`, `501`, `502`, `503`, or
      `504`) When you specify an explicit `negative_caching_policy`, ensure
      that you also specify a cache TTL for all response codes that you wish
      to cache. The CDNPolicy doesn't apply any default negative caching when
      a policy exists.
    signedRequestKeyset: Optional. The EdgeCacheKeyset containing the set of
      public keys used to validate signed requests at the edge. The following
      are both valid paths to an `EdgeCacheKeyset` resource: *
      `projects/project/locations/global/edgeCacheKeysets/yourKeyset` *
      `yourKeyset` SignedRequestMode must be set to a value other than
      DISABLED when a keyset is provided.
    signedRequestMaximumExpirationTtl: Optional. Limits how far into the
      future the expiration time of a signed request can be. When set, a
      signed request is rejected if its expiration time is later than `now` +
      `signed_request_maximum_expiration_ttl`, where `now` is the time at
      which the signed request is first handled by the CDN. - The TTL must be
      > 0. - Fractions of a second are not allowed. By default,
      `signed_request_maximum_expiration_ttl` is not set and the expiration
      time of a signed request might be arbitrarily far into future.
    signedRequestMode: Optional. Specifies whether to enforce signed requests.
      The default value is DISABLED, which means all content is public, and
      does not authorize access. You must also set a signed_request_keyset to
      enable signed requests. When set to REQUIRE_SIGNATURES or
      REQUIRE_TOKENS, all matching requests get their signature validated.
      Requests that aren't signed with the corresponding private key, or that
      are otherwise invalid (such as expired or do not match the signature, IP
      address, or header) are rejected with an HTTP 403 error. If logging is
      turned on, then invalid requests are also logged.
    signedTokenOptions: Optional. Any additional options for signed tokens.
      `signed_token_options` can only be specified when `signed_request_mode`
      is `REQUIRE_TOKENS`.
  """

    class CacheModeValueValuesEnum(_messages.Enum):
        """Optional. Set the CacheMode used by this route. BYPASS_CACHE and
    USE_ORIGIN_HEADERS proxy the origin's headers. Other cache modes pass
    Cache-Control to the client. Use client_ttl to override what is sent to
    the client.

    Values:
      CACHE_MODE_UNSPECIFIED: Unspecified value. Defaults to
        `CACHE_ALL_STATIC`.
      CACHE_ALL_STATIC: Automatically cache static content, including common
        image formats, media (video and audio), and web assets (JavaScript and
        CSS). Requests and responses that are marked as uncacheable, as well
        as dynamic content (including HTML), aren't cached.
      USE_ORIGIN_HEADERS: Only cache responses with valid HTTP caching
        directives. Responses without these headers aren't cached at Google's
        edge, and require a full trip to the origin on every request,
        potentially impacting performance and increasing load on the origin
        server.
      FORCE_CACHE_ALL: Cache all content, ignoring any `private`, `no-store`
        or `no-cache` directives in Cache-Control response headers.
        **Warning:** this might result in caching private, per-user (user
        identifiable) content. Only enable this on routes where the
        EdgeCacheOrigin doesn't serve private or dynamic content, such as
        storage buckets.
      BYPASS_CACHE: Bypass all caching for requests that match routes with
        this CDNPolicy applied. Enabling this causes the edge cache to ignore
        all HTTP caching directives. All responses are fulfilled from the
        origin.
    """
        CACHE_MODE_UNSPECIFIED = 0
        CACHE_ALL_STATIC = 1
        USE_ORIGIN_HEADERS = 2
        FORCE_CACHE_ALL = 3
        BYPASS_CACHE = 4

    class SignedRequestModeValueValuesEnum(_messages.Enum):
        """Optional. Specifies whether to enforce signed requests. The default
    value is DISABLED, which means all content is public, and does not
    authorize access. You must also set a signed_request_keyset to enable
    signed requests. When set to REQUIRE_SIGNATURES or REQUIRE_TOKENS, all
    matching requests get their signature validated. Requests that aren't
    signed with the corresponding private key, or that are otherwise invalid
    (such as expired or do not match the signature, IP address, or header) are
    rejected with an HTTP 403 error. If logging is turned on, then invalid
    requests are also logged.

    Values:
      SIGNED_REQUEST_MODE_UNSPECIFIED: Unspecified value. Defaults to
        `DISABLED`.
      DISABLED: Do not enforce signed requests.
      REQUIRE_SIGNATURES: Enforce signed requests using query parameter, path
        component, or cookie signatures. All requests must have a valid
        signature. Requests that are missing the signature (URL or cookie-
        based) are rejected as if the signature was invalid.
      REQUIRE_TOKENS: Enforce signed requests using signed tokens. All
        requests must have a valid signed token. Requests that are missing a
        signed token (URL or cookie-based) are rejected as if the signed token
        was invalid.
    """
        SIGNED_REQUEST_MODE_UNSPECIFIED = 0
        DISABLED = 1
        REQUIRE_SIGNATURES = 2
        REQUIRE_TOKENS = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NegativeCachingPolicyValue(_messages.Message):
        """Optional. A cache TTL for the specified HTTP status code.
    negative_caching must be enabled to configure `negative_caching_policy`.
    The following limitations apply: - Omitting the policy and leaving
    `negative_caching` enabled uses the default TTLs for each status code,
    defined in `negative_caching`. - TTLs must be >= `0` (where `0` is "always
    revalidate") and <= `86400s` (1 day) You can set only the following status
    codes: - HTTP redirection (`300`, `301`, `302`, `307`, or `308`) - Client
    error (`400`, `403`, `404`, `405`, `410`, `421`, or `451`) - Server error
    (`500`, `501`, `502`, `503`, or `504`) When you specify an explicit
    `negative_caching_policy`, ensure that you also specify a cache TTL for
    all response codes that you wish to cache. The CDNPolicy doesn't apply any
    default negative caching when a policy exists.

    Messages:
      AdditionalProperty: An additional property for a
        NegativeCachingPolicyValue object.

    Fields:
      additionalProperties: Additional properties of type
        NegativeCachingPolicyValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NegativeCachingPolicyValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    addSignatures = _messages.MessageField('CDNPolicyAddSignaturesOptions', 1)
    cacheKeyPolicy = _messages.MessageField('CDNPolicyCacheKeyPolicy', 2)
    cacheMode = _messages.EnumField('CacheModeValueValuesEnum', 3)
    clientTtl = _messages.StringField(4)
    defaultTtl = _messages.StringField(5)
    maxTtl = _messages.StringField(6)
    negativeCaching = _messages.BooleanField(7)
    negativeCachingPolicy = _messages.MessageField('NegativeCachingPolicyValue', 8)
    signedRequestKeyset = _messages.StringField(9)
    signedRequestMaximumExpirationTtl = _messages.StringField(10)
    signedRequestMode = _messages.EnumField('SignedRequestModeValueValuesEnum', 11)
    signedTokenOptions = _messages.MessageField('CDNPolicySignedTokenOptions', 12)