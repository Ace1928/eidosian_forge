from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetTcpProxy(_messages.Message):
    """Represents a Target TCP Proxy resource. A target TCP proxy is a
  component of a Proxy Network Load Balancer. The forwarding rule references
  the target TCP proxy, and the target proxy then references a backend
  service. For more information, read Proxy Network Load Balancer overview.

  Enums:
    ProxyHeaderValueValuesEnum: Specifies the type of proxy header to append
      before sending data to the backend, either NONE or PROXY_V1. The default
      is NONE.

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of the resource. Always compute#targetTcpProxy
      for target TCP proxies.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    proxyBind: This field only applies when the forwarding rule that
      references this target proxy has a loadBalancingScheme set to
      INTERNAL_SELF_MANAGED. When this field is set to true, Envoy proxies set
      up inbound traffic interception and bind to the IP address and port
      specified in the forwarding rule. This is generally useful when using
      Traffic Director to configure Envoy as a gateway or middle proxy (in
      other words, not a sidecar proxy). The Envoy proxy listens for inbound
      requests and handles requests when it receives them. The default is
      false.
    proxyHeader: Specifies the type of proxy header to append before sending
      data to the backend, either NONE or PROXY_V1. The default is NONE.
    region: [Output Only] URL of the region where the regional TCP proxy
      resides. This field is not applicable to global TCP proxy.
    selfLink: [Output Only] Server-defined URL for the resource.
    service: URL to the BackendService resource.
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
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    id = _messages.IntegerField(3, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(4, default='compute#targetTcpProxy')
    name = _messages.StringField(5)
    proxyBind = _messages.BooleanField(6)
    proxyHeader = _messages.EnumField('ProxyHeaderValueValuesEnum', 7)
    region = _messages.StringField(8)
    selfLink = _messages.StringField(9)
    service = _messages.StringField(10)