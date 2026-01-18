from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgeCacheService(_messages.Message):
    """Defines the IP addresses, protocols, security policies, cache policies,
  and routing configuration.

  Messages:
    LabelsValue: Optional. A set of label tags associated with the
      `EdgeCacheService` resource.

  Fields:
    createTime: Output only. The creation timestamp in RFC3339 text format.
    description: Optional. A human-readable description of the resource.
    disableHttp2: Optional. Disables HTTP/2. HTTP/2 (h2) is enabled by default
      and recommended for performance. HTTP/2 improves connection re-use and
      reduces connection setup overhead by sending multiple streams over the
      same connection. Some legacy HTTP clients might have issues with HTTP/2
      connections due to broken HTTP/2 implementations. Setting this to `true`
      prevents HTTP/2 from being advertised and negotiated.
    disableQuic: Optional. HTTP/3 (IETF QUIC) and Google QUIC are enabled by
      default.
    edgeSecurityPolicy: Optional. The resource URL that points at the Cloud
      Armor edge security policy that is applied on each request against the
      EdgeCacheService resource.
    edgeSslCertificates: Optional. Certificate resources that are used to
      authenticate connections between users and the EdgeCacheService
      resource. Note that only global certificates with a scope of
      `EDGE_CACHE` can be attached to an EdgeCacheService resource. The
      following are both valid paths to an `edge_ssl_certificates` resource: *
      `projects/project/locations/global/certificates/media-example-com-cert`
      * `media-example-com-cert` You can specify up to five SSL certificates.
    ipv4Addresses: Output only. The IPv4 addresses associated with this
      service. Addresses are static for the lifetime of the service. IP
      addresses provisioned via Bring-Your-Own-IP (BYOIP) are not supported.
    ipv6Addresses: Output only. The IPv6 addresses associated with this
      service. Addresses are static for the lifetime of the service. IP
      addresses provisioned via Bring-Your-Own-IP (BYOIP) are not supported.
    labels: Optional. A set of label tags associated with the
      `EdgeCacheService` resource.
    logConfig: Optional. The logging options for the traffic served by this
      service. If logging is enabled, logs are exported to Cloud Logging.
    name: Required. The name of the resource as provided by the client when
      the resource is created. The name must be 1-64 characters long, and
      match the regular expression `[a-zA-Z]([a-zA-Z0-9_-])*` which means the
      first character must be a letter, and all following characters must be a
      dash, an underscore, a letter, or a digit.
    requireTls: Optional. Require TLS (HTTPS) for all clients connecting to
      this service. Clients who connect over HTTP (port 80) see an `HTTP 301`
      response to the same URL over HTTPS (port 443). You must have at least
      one edge_ssl_certificates specified to enable this.
    routing: Required. Defines how requests are routed, modified, and cached,
      and which origin the content is filled from.
    updateTime: Output only. The update timestamp in RFC3339 text format.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. A set of label tags associated with the `EdgeCacheService`
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
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    disableHttp2 = _messages.BooleanField(3)
    disableQuic = _messages.BooleanField(4)
    edgeSecurityPolicy = _messages.StringField(5)
    edgeSslCertificates = _messages.StringField(6, repeated=True)
    ipv4Addresses = _messages.StringField(7, repeated=True)
    ipv6Addresses = _messages.StringField(8, repeated=True)
    labels = _messages.MessageField('LabelsValue', 9)
    logConfig = _messages.MessageField('LogConfig', 10)
    name = _messages.StringField(11)
    requireTls = _messages.BooleanField(12)
    routing = _messages.MessageField('Routing', 13)
    updateTime = _messages.StringField(14)