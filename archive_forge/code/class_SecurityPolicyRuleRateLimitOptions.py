from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleRateLimitOptions(_messages.Message):
    """A SecurityPolicyRuleRateLimitOptions object.

  Enums:
    EnforceOnKeyValueValuesEnum: Determines the key to enforce the
      rate_limit_threshold on. Possible values are: - ALL: A single rate limit
      threshold is applied to all the requests matching this rule. This is the
      default value if "enforceOnKey" is not configured. - IP: The source IP
      address of the request is the key. Each IP has this limit enforced
      separately. - HTTP_HEADER: The value of the HTTP header whose name is
      configured under "enforceOnKeyName". The key value is truncated to the
      first 128 bytes of the header value. If no such header is present in the
      request, the key type defaults to ALL. - XFF_IP: The first IP address
      (i.e. the originating client IP address) specified in the list of IPs
      under X-Forwarded-For HTTP header. If no such header is present or the
      value is not a valid IP, the key defaults to the source IP address of
      the request i.e. key type IP. - HTTP_COOKIE: The value of the HTTP
      cookie whose name is configured under "enforceOnKeyName". The key value
      is truncated to the first 128 bytes of the cookie value. If no such
      cookie is present in the request, the key type defaults to ALL. -
      HTTP_PATH: The URL path of the HTTP request. The key value is truncated
      to the first 128 bytes. - SNI: Server name indication in the TLS session
      of the HTTPS request. The key value is truncated to the first 128 bytes.
      The key type defaults to ALL on a HTTP session. - REGION_CODE: The
      country/region from which the request originates. - TLS_JA3_FINGERPRINT:
      JA3 TLS/SSL fingerprint if the client connects using HTTPS, HTTP/2 or
      HTTP/3. If not available, the key type defaults to ALL. - USER_IP: The
      IP address of the originating client, which is resolved based on
      "userIpRequestHeaders" configured with the security policy. If there is
      no "userIpRequestHeaders" configuration or an IP address cannot be
      resolved from it, the key type defaults to IP.

  Fields:
    banDurationSec: Can only be specified if the action for the rule is
      "rate_based_ban". If specified, determines the time (in seconds) the
      traffic will continue to be banned by the rate limit after the rate
      falls below the threshold.
    banThreshold: Can only be specified if the action for the rule is
      "rate_based_ban". If specified, the key will be banned for the
      configured 'ban_duration_sec' when the number of requests that exceed
      the 'rate_limit_threshold' also exceed this 'ban_threshold'.
    conformAction: Action to take for requests that are under the configured
      rate limit threshold. Valid option is "allow" only.
    enforceOnKey: Determines the key to enforce the rate_limit_threshold on.
      Possible values are: - ALL: A single rate limit threshold is applied to
      all the requests matching this rule. This is the default value if
      "enforceOnKey" is not configured. - IP: The source IP address of the
      request is the key. Each IP has this limit enforced separately. -
      HTTP_HEADER: The value of the HTTP header whose name is configured under
      "enforceOnKeyName". The key value is truncated to the first 128 bytes of
      the header value. If no such header is present in the request, the key
      type defaults to ALL. - XFF_IP: The first IP address (i.e. the
      originating client IP address) specified in the list of IPs under
      X-Forwarded-For HTTP header. If no such header is present or the value
      is not a valid IP, the key defaults to the source IP address of the
      request i.e. key type IP. - HTTP_COOKIE: The value of the HTTP cookie
      whose name is configured under "enforceOnKeyName". The key value is
      truncated to the first 128 bytes of the cookie value. If no such cookie
      is present in the request, the key type defaults to ALL. - HTTP_PATH:
      The URL path of the HTTP request. The key value is truncated to the
      first 128 bytes. - SNI: Server name indication in the TLS session of the
      HTTPS request. The key value is truncated to the first 128 bytes. The
      key type defaults to ALL on a HTTP session. - REGION_CODE: The
      country/region from which the request originates. - TLS_JA3_FINGERPRINT:
      JA3 TLS/SSL fingerprint if the client connects using HTTPS, HTTP/2 or
      HTTP/3. If not available, the key type defaults to ALL. - USER_IP: The
      IP address of the originating client, which is resolved based on
      "userIpRequestHeaders" configured with the security policy. If there is
      no "userIpRequestHeaders" configuration or an IP address cannot be
      resolved from it, the key type defaults to IP.
    enforceOnKeyConfigs: If specified, any combination of values of
      enforce_on_key_type/enforce_on_key_name is treated as the key on which
      ratelimit threshold/action is enforced. You can specify up to 3
      enforce_on_key_configs. If enforce_on_key_configs is specified,
      enforce_on_key must not be specified.
    enforceOnKeyName: Rate limit key name applicable only for the following
      key types: HTTP_HEADER -- Name of the HTTP header whose value is taken
      as the key value. HTTP_COOKIE -- Name of the HTTP cookie whose value is
      taken as the key value.
    exceedAction: Action to take for requests that are above the configured
      rate limit threshold, to either deny with a specified HTTP response
      code, or redirect to a different endpoint. Valid options are
      `deny(STATUS)`, where valid values for `STATUS` are 403, 404, 429, and
      502, and `redirect`, where the redirect parameters come from
      `exceedRedirectOptions` below. The `redirect` action is only supported
      in Global Security Policies of type CLOUD_ARMOR.
    exceedRedirectOptions: Parameters defining the redirect action that is
      used as the exceed action. Cannot be specified if the exceed action is
      not redirect. This field is only supported in Global Security Policies
      of type CLOUD_ARMOR.
    rateLimitThreshold: Threshold at which to begin ratelimiting.
  """

    class EnforceOnKeyValueValuesEnum(_messages.Enum):
        """Determines the key to enforce the rate_limit_threshold on. Possible
    values are: - ALL: A single rate limit threshold is applied to all the
    requests matching this rule. This is the default value if "enforceOnKey"
    is not configured. - IP: The source IP address of the request is the key.
    Each IP has this limit enforced separately. - HTTP_HEADER: The value of
    the HTTP header whose name is configured under "enforceOnKeyName". The key
    value is truncated to the first 128 bytes of the header value. If no such
    header is present in the request, the key type defaults to ALL. - XFF_IP:
    The first IP address (i.e. the originating client IP address) specified in
    the list of IPs under X-Forwarded-For HTTP header. If no such header is
    present or the value is not a valid IP, the key defaults to the source IP
    address of the request i.e. key type IP. - HTTP_COOKIE: The value of the
    HTTP cookie whose name is configured under "enforceOnKeyName". The key
    value is truncated to the first 128 bytes of the cookie value. If no such
    cookie is present in the request, the key type defaults to ALL. -
    HTTP_PATH: The URL path of the HTTP request. The key value is truncated to
    the first 128 bytes. - SNI: Server name indication in the TLS session of
    the HTTPS request. The key value is truncated to the first 128 bytes. The
    key type defaults to ALL on a HTTP session. - REGION_CODE: The
    country/region from which the request originates. - TLS_JA3_FINGERPRINT:
    JA3 TLS/SSL fingerprint if the client connects using HTTPS, HTTP/2 or
    HTTP/3. If not available, the key type defaults to ALL. - USER_IP: The IP
    address of the originating client, which is resolved based on
    "userIpRequestHeaders" configured with the security policy. If there is no
    "userIpRequestHeaders" configuration or an IP address cannot be resolved
    from it, the key type defaults to IP.

    Values:
      ALL: <no description>
      ALL_IPS: <no description>
      HTTP_COOKIE: <no description>
      HTTP_HEADER: <no description>
      HTTP_PATH: <no description>
      IP: <no description>
      REGION_CODE: <no description>
      SNI: <no description>
      TLS_JA3_FINGERPRINT: <no description>
      USER_IP: <no description>
      XFF_IP: <no description>
    """
        ALL = 0
        ALL_IPS = 1
        HTTP_COOKIE = 2
        HTTP_HEADER = 3
        HTTP_PATH = 4
        IP = 5
        REGION_CODE = 6
        SNI = 7
        TLS_JA3_FINGERPRINT = 8
        USER_IP = 9
        XFF_IP = 10
    banDurationSec = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    banThreshold = _messages.MessageField('SecurityPolicyRuleRateLimitOptionsThreshold', 2)
    conformAction = _messages.StringField(3)
    enforceOnKey = _messages.EnumField('EnforceOnKeyValueValuesEnum', 4)
    enforceOnKeyConfigs = _messages.MessageField('SecurityPolicyRuleRateLimitOptionsEnforceOnKeyConfig', 5, repeated=True)
    enforceOnKeyName = _messages.StringField(6)
    exceedAction = _messages.StringField(7)
    exceedRedirectOptions = _messages.MessageField('SecurityPolicyRuleRedirectOptions', 8)
    rateLimitThreshold = _messages.MessageField('SecurityPolicyRuleRateLimitOptionsThreshold', 9)