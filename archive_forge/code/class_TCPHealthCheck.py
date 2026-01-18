from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TCPHealthCheck(_messages.Message):
    """A TCPHealthCheck object.

  Enums:
    PortSpecificationValueValuesEnum: Specifies how a port is selected for
      health checking. Can be one of the following values: USE_FIXED_PORT:
      Specifies a port number explicitly using the port field in the health
      check. Supported by backend services for passthrough load balancers and
      backend services for proxy load balancers. Not supported by target
      pools. The health check supports all backends supported by the backend
      service provided the backend can be health checked. For example,
      GCE_VM_IP network endpoint groups, GCE_VM_IP_PORT network endpoint
      groups, and instance group backends. USE_NAMED_PORT: Not supported.
      USE_SERVING_PORT: Provides an indirect method of specifying the health
      check port by referring to the backend service. Only supported by
      backend services for proxy load balancers. Not supported by target
      pools. Not supported by backend services for passthrough load balancers.
      Supports all backends that can be health checked; for example,
      GCE_VM_IP_PORT network endpoint groups and instance group backends. For
      GCE_VM_IP_PORT network endpoint group backends, the health check uses
      the port number specified for each endpoint in the network endpoint
      group. For instance group backends, the health check uses the port
      number determined by looking up the backend service's named port in the
      instance group's list of named ports.
    ProxyHeaderValueValuesEnum: Specifies the type of proxy header to append
      before sending data to the backend, either NONE or PROXY_V1. The default
      is NONE.

  Fields:
    port: The TCP port number to which the health check prober sends packets.
      The default value is 80. Valid values are 1 through 65535.
    portName: Not supported.
    portSpecification: Specifies how a port is selected for health checking.
      Can be one of the following values: USE_FIXED_PORT: Specifies a port
      number explicitly using the port field in the health check. Supported by
      backend services for passthrough load balancers and backend services for
      proxy load balancers. Not supported by target pools. The health check
      supports all backends supported by the backend service provided the
      backend can be health checked. For example, GCE_VM_IP network endpoint
      groups, GCE_VM_IP_PORT network endpoint groups, and instance group
      backends. USE_NAMED_PORT: Not supported. USE_SERVING_PORT: Provides an
      indirect method of specifying the health check port by referring to the
      backend service. Only supported by backend services for proxy load
      balancers. Not supported by target pools. Not supported by backend
      services for passthrough load balancers. Supports all backends that can
      be health checked; for example, GCE_VM_IP_PORT network endpoint groups
      and instance group backends. For GCE_VM_IP_PORT network endpoint group
      backends, the health check uses the port number specified for each
      endpoint in the network endpoint group. For instance group backends, the
      health check uses the port number determined by looking up the backend
      service's named port in the instance group's list of named ports.
    proxyHeader: Specifies the type of proxy header to append before sending
      data to the backend, either NONE or PROXY_V1. The default is NONE.
    request: Instructs the health check prober to send this exact ASCII
      string, up to 1024 bytes in length, after establishing the TCP
      connection.
    response: Creates a content-based TCP health check. In addition to
      establishing a TCP connection, you can configure the health check to
      pass only when the backend sends this exact response ASCII string, up to
      1024 bytes in length. For details, see: https://cloud.google.com/load-
      balancing/docs/health-check-concepts#criteria-protocol-ssl-tcp
  """

    class PortSpecificationValueValuesEnum(_messages.Enum):
        """Specifies how a port is selected for health checking. Can be one of
    the following values: USE_FIXED_PORT: Specifies a port number explicitly
    using the port field in the health check. Supported by backend services
    for passthrough load balancers and backend services for proxy load
    balancers. Not supported by target pools. The health check supports all
    backends supported by the backend service provided the backend can be
    health checked. For example, GCE_VM_IP network endpoint groups,
    GCE_VM_IP_PORT network endpoint groups, and instance group backends.
    USE_NAMED_PORT: Not supported. USE_SERVING_PORT: Provides an indirect
    method of specifying the health check port by referring to the backend
    service. Only supported by backend services for proxy load balancers. Not
    supported by target pools. Not supported by backend services for
    passthrough load balancers. Supports all backends that can be health
    checked; for example, GCE_VM_IP_PORT network endpoint groups and instance
    group backends. For GCE_VM_IP_PORT network endpoint group backends, the
    health check uses the port number specified for each endpoint in the
    network endpoint group. For instance group backends, the health check uses
    the port number determined by looking up the backend service's named port
    in the instance group's list of named ports.

    Values:
      USE_FIXED_PORT: The port number in the health check's port is used for
        health checking. Applies to network endpoint group and instance group
        backends.
      USE_NAMED_PORT: Not supported.
      USE_SERVING_PORT: For network endpoint group backends, the health check
        uses the port number specified on each endpoint in the network
        endpoint group. For instance group backends, the health check uses the
        port number specified for the backend service's named port defined in
        the instance group's named ports.
    """
        USE_FIXED_PORT = 0
        USE_NAMED_PORT = 1
        USE_SERVING_PORT = 2

    class ProxyHeaderValueValuesEnum(_messages.Enum):
        """Specifies the type of proxy header to append before sending data to
    the backend, either NONE or PROXY_V1. The default is NONE.

    Values:
      NONE: <no description>
      PROXY_V1: <no description>
    """
        NONE = 0
        PROXY_V1 = 1
    port = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    portName = _messages.StringField(2)
    portSpecification = _messages.EnumField('PortSpecificationValueValuesEnum', 3)
    proxyHeader = _messages.EnumField('ProxyHeaderValueValuesEnum', 4)
    request = _messages.StringField(5)
    response = _messages.StringField(6)