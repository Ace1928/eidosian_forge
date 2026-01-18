from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def HandlePortRelatedFlagsForUpdate(args, x_health_check):
    """Calculate port, port_name and port_specification for HC update."""
    port = x_health_check.port
    port_name = x_health_check.portName
    port_specification = x_health_check.portSpecification
    enum_class = type(x_health_check).PortSpecificationValueValuesEnum
    if args.use_serving_port:
        if args.IsSpecified('port_name'):
            _RaiseBadPortSpecificationError('--port-name', '--use-serving-port', '--use-serving-port')
        if args.IsSpecified('port'):
            _RaiseBadPortSpecificationError('--port', '--use-serving-port', '--use-serving-port')
        port = None
        port_name = None
        port_specification = enum_class.USE_SERVING_PORT
    if args.IsSpecified('port'):
        port = args.port
        port_name = None
        port_specification = enum_class.USE_FIXED_PORT
    elif args.IsSpecified('port_name'):
        if args.port_name:
            port = None
            port_name = args.port_name
            port_specification = enum_class.USE_NAMED_PORT
        else:
            port_name = None
            port_specification = enum_class.USE_FIXED_PORT
    else:
        pass
    return (port, port_name, port_specification)