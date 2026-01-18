from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServerTlsPolicy(_messages.Message):
    """ServerTlsPolicy is a resource that specifies how a server should
  authenticate incoming requests. This resource itself does not affect
  configuration unless it is attached to a target HTTPS proxy or endpoint
  config selector resource. ServerTlsPolicy in the form accepted by external
  HTTPS load balancers can be attached only to TargetHttpsProxy with an
  `EXTERNAL` or `EXTERNAL_MANAGED` load balancing scheme. Traffic Director
  compatible ServerTlsPolicies can be attached to EndpointPolicy and
  TargetHttpsProxy with Traffic Director `INTERNAL_SELF_MANAGED` load
  balancing scheme.

  Enums:
    MaxTlsVersionValueValuesEnum: Optional. TLS max version used only for
      Envoy. If not specified, Envoy will use default version. Envoy latest: h
      ttps://www.envoyproxy.io/docs/envoy/latest/api-
      v3/extensions/transport_sockets/tls/v3/common.proto
    MinTlsVersionValueValuesEnum: Optional. TLS min version used only for
      Envoy. If not specified, Envoy will use default version. Envoy latest: h
      ttps://www.envoyproxy.io/docs/envoy/latest/api-
      v3/extensions/transport_sockets/tls/v3/common.proto

  Messages:
    LabelsValue: Set of label tags associated with the resource.

  Fields:
    allowOpen: This field applies only for Traffic Director policies. It is
      must be set to false for external HTTPS load balancer policies.
      Determines if server allows plaintext connections. If set to true,
      server allows plain text connections. By default, it is set to false.
      This setting is not exclusive of other encryption modes. For example, if
      `allow_open` and `mtls_policy` are set, server allows both plain text
      and mTLS connections. See documentation of other encryption modes to
      confirm compatibility. Consider using it if you wish to upgrade in place
      your deployment to TLS while having mixed TLS and non-TLS traffic
      reaching port :80.
    cipherSuites: Optional. TLS custom cipher suites used only in CSM.
      Following ciphers are supported: ECDHE-ECDSA-AES128-GCM-SHA256 ECDHE-
      RSA-AES128-GCM-SHA256 ECDHE-ECDSA-AES256-GCM-SHA384 ECDHE-RSA-
      AES256-GCM-SHA384 ECDHE-ECDSA-CHACHA20-POLY1305 ECDHE-RSA-
      CHACHA20-POLY1305 ECDHE-ECDSA-AES128-SHA ECDHE-RSA-AES128-SHA ECDHE-
      ECDSA-AES256-SHA ECDHE-RSA-AES256-SHA AES128-GCM-SHA256 AES256-GCM-
      SHA384 AES128-SHA AES256-SHA DES-CBC3-SHA
    createTime: Output only. The timestamp when the resource was created.
    description: Free-text description of the resource.
    internalCaller: Optional. A flag set to identify internal controllers
      Setting this will trigger a P4SA check to validate the caller is from an
      allowlisted service's P4SA even if other optional fields are unset.
    labels: Set of label tags associated with the resource.
    maxTlsVersion: Optional. TLS max version used only for Envoy. If not
      specified, Envoy will use default version. Envoy latest: https://www.env
      oyproxy.io/docs/envoy/latest/api-
      v3/extensions/transport_sockets/tls/v3/common.proto
    minTlsVersion: Optional. TLS min version used only for Envoy. If not
      specified, Envoy will use default version. Envoy latest: https://www.env
      oyproxy.io/docs/envoy/latest/api-
      v3/extensions/transport_sockets/tls/v3/common.proto
    mtlsPolicy: This field is required if the policy is used with external
      HTTPS load balancers. This field can be empty for Traffic Director.
      Defines a mechanism to provision peer validation certificates for peer
      to peer authentication (Mutual TLS - mTLS). If not specified, client
      certificate will not be requested. The connection is treated as TLS and
      not mTLS. If `allow_open` and `mtls_policy` are set, server allows both
      plain text and mTLS connections.
    name: Required. Name of the ServerTlsPolicy resource. It matches the
      pattern
      `projects/*/locations/{location}/serverTlsPolicies/{server_tls_policy}`
    serverCertificate: Optional if policy is to be used with Traffic Director.
      For external HTTPS load balancer must be empty. Defines a mechanism to
      provision server identity (public and private keys). Cannot be combined
      with `allow_open` as a permissive mode that allows both plain text and
      TLS is not supported.
    subjectAltNames: Optional. Server side validation for client SAN, only
      used in CSM. If not specified, the client SAN will not be checked by the
      server.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    class MaxTlsVersionValueValuesEnum(_messages.Enum):
        """Optional. TLS max version used only for Envoy. If not specified, Envoy
    will use default version. Envoy latest: https://www.envoyproxy.io/docs/env
    oy/latest/api-v3/extensions/transport_sockets/tls/v3/common.proto

    Values:
      TLS_VERSION_UNSPECIFIED: <no description>
      TLS_V1_0: <no description>
      TLS_V1_1: <no description>
      TLS_V1_2: <no description>
      TLS_V1_3: <no description>
    """
        TLS_VERSION_UNSPECIFIED = 0
        TLS_V1_0 = 1
        TLS_V1_1 = 2
        TLS_V1_2 = 3
        TLS_V1_3 = 4

    class MinTlsVersionValueValuesEnum(_messages.Enum):
        """Optional. TLS min version used only for Envoy. If not specified, Envoy
    will use default version. Envoy latest: https://www.envoyproxy.io/docs/env
    oy/latest/api-v3/extensions/transport_sockets/tls/v3/common.proto

    Values:
      TLS_VERSION_UNSPECIFIED: <no description>
      TLS_V1_0: <no description>
      TLS_V1_1: <no description>
      TLS_V1_2: <no description>
      TLS_V1_3: <no description>
    """
        TLS_VERSION_UNSPECIFIED = 0
        TLS_V1_0 = 1
        TLS_V1_1 = 2
        TLS_V1_2 = 3
        TLS_V1_3 = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Set of label tags associated with the resource.

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
    allowOpen = _messages.BooleanField(1)
    cipherSuites = _messages.StringField(2, repeated=True)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    internalCaller = _messages.BooleanField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    maxTlsVersion = _messages.EnumField('MaxTlsVersionValueValuesEnum', 7)
    minTlsVersion = _messages.EnumField('MinTlsVersionValueValuesEnum', 8)
    mtlsPolicy = _messages.MessageField('MTLSPolicy', 9)
    name = _messages.StringField(10)
    serverCertificate = _messages.MessageField('GoogleCloudNetworksecurityV1CertificateProvider', 11)
    subjectAltNames = _messages.StringField(12, repeated=True)
    updateTime = _messages.StringField(13)