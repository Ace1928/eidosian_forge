from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def ValidateAndAddPortSpecificationToHealthCheck(args, x_health_check):
    """Modifies the health check as needed and adds port spec to the check."""
    if args.IsSpecified('port_name') and (not args.IsSpecified('port')):
        x_health_check.port = None
    enum_class = type(x_health_check).PortSpecificationValueValuesEnum
    if args.use_serving_port:
        if args.IsSpecified('port_name'):
            _RaiseBadPortSpecificationError('--port-name', '--use-serving-port', '--use-serving-port')
        if args.IsSpecified('port'):
            _RaiseBadPortSpecificationError('--port', '--use-serving-port', '--use-serving-port')
        x_health_check.portSpecification = enum_class.USE_SERVING_PORT
        x_health_check.port = None
    elif args.IsSpecified('port') and args.IsSpecified('port_name'):
        x_health_check.portSpecification = enum_class.USE_FIXED_PORT
        x_health_check.portName = None
    elif args.IsSpecified('port_name'):
        x_health_check.portSpecification = enum_class.USE_NAMED_PORT
    else:
        x_health_check.portSpecification = enum_class.USE_FIXED_PORT