from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GRPCTLSHealthCheck(_messages.Message):
    """A GRPCTLSHealthCheck object.

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

  Fields:
    grpcServiceName: The gRPC service name for the health check. This field is
      optional. The value of grpc_service_name has the following meanings by
      convention: - Empty service_name means the overall status of all
      services at the backend. - Non-empty service_name means the health of
      that gRPC service, as defined by the owner of the service. The
      grpc_service_name can only be ASCII.
    port: The TCP port number to which the health check prober sends packets.
      Valid values are 1 through 65535.
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
    grpcServiceName = _messages.StringField(1)
    port = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    portSpecification = _messages.EnumField('PortSpecificationValueValuesEnum', 3)