from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetSslProxy(_messages.Message):
    """Represents a Target SSL Proxy resource. A target SSL proxy is a
  component of a Proxy Network Load Balancer. The forwarding rule references
  the target SSL proxy, and the target proxy then references a backend
  service. For more information, read Proxy Network Load Balancer overview.

  Enums:
    ProxyHeaderValueValuesEnum: Specifies the type of proxy header to append
      before sending data to the backend, either NONE or PROXY_V1. The default
      is NONE.

  Fields:
    certificateMap: URL of a certificate map that identifies a certificate map
      associated with the given target proxy. This field can only be set for
      global target proxies. If set, sslCertificates will be ignored. Accepted
      format is //certificatemanager.googleapis.com/projects/{project
      }/locations/{location}/certificateMaps/{resourceName}.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of the resource. Always compute#targetSslProxy
      for target SSL proxies.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    proxyHeader: Specifies the type of proxy header to append before sending
      data to the backend, either NONE or PROXY_V1. The default is NONE.
    selfLink: [Output Only] Server-defined URL for the resource.
    service: URL to the BackendService resource.
    sslCertificates: URLs to SslCertificate resources that are used to
      authenticate connections to Backends. At least one SSL certificate must
      be specified. Currently, you may specify up to 15 SSL certificates.
      sslCertificates do not apply when the load balancing scheme is set to
      INTERNAL_SELF_MANAGED.
    sslPolicy: URL of SslPolicy resource that will be associated with the
      TargetSslProxy resource. If not set, the TargetSslProxy resource will
      not have any SSL policy configured.
  """

    class ProxyHeaderValueValuesEnum(_messages.Enum):
        """Specifies the type of proxy header to append before sending data to
    the backend, either NONE or PROXY_V1. The default is NONE.

    Values:
      NONE: <no description>
      PROXY_V1: <no description>
    """
        NONE = 0
        PROXY_V1 = 1
    certificateMap = _messages.StringField(1)
    creationTimestamp = _messages.StringField(2)
    description = _messages.StringField(3)
    id = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(5, default='compute#targetSslProxy')
    name = _messages.StringField(6)
    proxyHeader = _messages.EnumField('ProxyHeaderValueValuesEnum', 7)
    selfLink = _messages.StringField(8)
    service = _messages.StringField(9)
    sslCertificates = _messages.StringField(10, repeated=True)
    sslPolicy = _messages.StringField(11)