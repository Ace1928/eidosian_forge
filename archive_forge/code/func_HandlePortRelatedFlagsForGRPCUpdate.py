from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def HandlePortRelatedFlagsForGRPCUpdate(args, x_health_check):
    """Calculate port and port_specification for gRPC HC update."""
    port = x_health_check.port
    port_specification = x_health_check.portSpecification
    enum_class = type(x_health_check).PortSpecificationValueValuesEnum
    if args.use_serving_port:
        if args.IsSpecified('port'):
            _RaiseBadPortSpecificationError('--port', '--use-serving-port', '--use-serving-port')
        port = None
        port_specification = enum_class.USE_SERVING_PORT
    if args.IsSpecified('port'):
        port = args.port
        port_specification = enum_class.USE_FIXED_PORT
    else:
        pass
    return (port, port_specification)