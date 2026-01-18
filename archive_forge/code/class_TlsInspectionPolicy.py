from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsInspectionPolicy(_messages.Message):
    """The TlsInspectionPolicy resource contains references to CA pools in
  Certificate Authority Service and associated metadata.

  Enums:
    MinTlsVersionValueValuesEnum: Optional. Minimum TLS version that the
      firewall should use when negotiating connections with both clients and
      servers. If this is not set, then the default value is to allow the
      broadest set of clients and servers (TLS 1.0 or higher). Setting this to
      more restrictive values may improve security, but may also prevent the
      firewall from connecting to some clients or servers. Note that Secure
      Web Proxy does not yet honor this field.
    TlsFeatureProfileValueValuesEnum: Optional. The selected Profile. If this
      is not set, then the default value is to allow the broadest set of
      clients and servers ("PROFILE_COMPATIBLE"). Setting this to more
      restrictive values may improve security, but may also prevent the TLS
      inspection proxy from connecting to some clients or servers. Note that
      Secure Web Proxy does not yet honor this field.

  Fields:
    caPool: Required. A CA pool resource used to issue interception
      certificates. The CA pool string has a relative resource path following
      the form "projects/{project}/locations/{location}/caPools/{ca_pool}".
    createTime: Output only. The timestamp when the resource was created.
    customTlsFeatures: Optional. List of custom TLS cipher suites selected.
      This field is valid only if the selected tls_feature_profile is CUSTOM.
      The compute.SslPoliciesService.ListAvailableFeatures method returns the
      set of features that can be specified in this list. Note that Secure Web
      Proxy does not yet honor this field.
    description: Optional. Free-text description of the resource.
    excludePublicCaSet: Optional. If FALSE (the default), use our default set
      of public CAs in addition to any CAs specified in trust_config. These
      public CAs are currently based on the Mozilla Root Program and are
      subject to change over time. If TRUE, do not accept our default set of
      public CAs. Only CAs specified in trust_config will be accepted. This
      defaults to FALSE (use public CAs in addition to trust_config) for
      backwards compatibility, but trusting public root CAs is *not
      recommended* unless the traffic in question is outbound to public web
      servers. When possible, prefer setting this to "false" and explicitly
      specifying trusted CAs and certificates in a TrustConfig. Note that
      Secure Web Proxy does not yet honor this field.
    minTlsVersion: Optional. Minimum TLS version that the firewall should use
      when negotiating connections with both clients and servers. If this is
      not set, then the default value is to allow the broadest set of clients
      and servers (TLS 1.0 or higher). Setting this to more restrictive values
      may improve security, but may also prevent the firewall from connecting
      to some clients or servers. Note that Secure Web Proxy does not yet
      honor this field.
    name: Required. Name of the resource. Name is of the form projects/{projec
      t}/locations/{location}/tlsInspectionPolicies/{tls_inspection_policy}
      tls_inspection_policy should match the
      pattern:(^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$).
    tlsFeatureProfile: Optional. The selected Profile. If this is not set,
      then the default value is to allow the broadest set of clients and
      servers ("PROFILE_COMPATIBLE"). Setting this to more restrictive values
      may improve security, but may also prevent the TLS inspection proxy from
      connecting to some clients or servers. Note that Secure Web Proxy does
      not yet honor this field.
    trustConfig: Optional. A TrustConfig resource used when making a
      connection to the TLS server. This is a relative resource path following
      the form
      "projects/{project}/locations/{location}/trustConfigs/{trust_config}".
      This is necessary to intercept TLS connections to servers with
      certificates signed by a private CA or self-signed certificates. Note
      that Secure Web Proxy does not yet honor this field.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    class MinTlsVersionValueValuesEnum(_messages.Enum):
        """Optional. Minimum TLS version that the firewall should use when
    negotiating connections with both clients and servers. If this is not set,
    then the default value is to allow the broadest set of clients and servers
    (TLS 1.0 or higher). Setting this to more restrictive values may improve
    security, but may also prevent the firewall from connecting to some
    clients or servers. Note that Secure Web Proxy does not yet honor this
    field.

    Values:
      TLS_VERSION_UNSPECIFIED: Indicates no TLS version was specified.
      TLS_1_0: TLS 1.0
      TLS_1_1: TLS 1.1
      TLS_1_2: TLS 1.2
      TLS_1_3: TLS 1.3
    """
        TLS_VERSION_UNSPECIFIED = 0
        TLS_1_0 = 1
        TLS_1_1 = 2
        TLS_1_2 = 3
        TLS_1_3 = 4

    class TlsFeatureProfileValueValuesEnum(_messages.Enum):
        """Optional. The selected Profile. If this is not set, then the default
    value is to allow the broadest set of clients and servers
    ("PROFILE_COMPATIBLE"). Setting this to more restrictive values may
    improve security, but may also prevent the TLS inspection proxy from
    connecting to some clients or servers. Note that Secure Web Proxy does not
    yet honor this field.

    Values:
      PROFILE_UNSPECIFIED: Indicates no profile was specified.
      PROFILE_COMPATIBLE: Compatible profile. Allows the broadest set of
        clients, even those which support only out-of-date SSL features to
        negotiate with the TLS inspection proxy.
      PROFILE_MODERN: Modern profile. Supports a wide set of SSL features,
        allowing modern clients to negotiate SSL with the TLS inspection
        proxy.
      PROFILE_RESTRICTED: Restricted profile. Supports a reduced set of SSL
        features, intended to meet stricter compliance requirements.
      PROFILE_CUSTOM: Custom profile. Allow only the set of allowed SSL
        features specified in the custom_features field of SslPolicy.
    """
        PROFILE_UNSPECIFIED = 0
        PROFILE_COMPATIBLE = 1
        PROFILE_MODERN = 2
        PROFILE_RESTRICTED = 3
        PROFILE_CUSTOM = 4
    caPool = _messages.StringField(1)
    createTime = _messages.StringField(2)
    customTlsFeatures = _messages.StringField(3, repeated=True)
    description = _messages.StringField(4)
    excludePublicCaSet = _messages.BooleanField(5)
    minTlsVersion = _messages.EnumField('MinTlsVersionValueValuesEnum', 6)
    name = _messages.StringField(7)
    tlsFeatureProfile = _messages.EnumField('TlsFeatureProfileValueValuesEnum', 8)
    trustConfig = _messages.StringField(9)
    updateTime = _messages.StringField(10)