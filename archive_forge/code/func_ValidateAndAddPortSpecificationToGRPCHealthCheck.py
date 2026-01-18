from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def ValidateAndAddPortSpecificationToGRPCHealthCheck(args, x_health_check):
    """Modifies the gRPC health check as needed and adds port specification."""
    enum_class = type(x_health_check).PortSpecificationValueValuesEnum
    if args.use_serving_port:
        if args.IsSpecified('port'):
            _RaiseBadPortSpecificationError('--port', '--use-serving-port', '--use-serving-port')
        x_health_check.portSpecification = enum_class.USE_SERVING_PORT
        x_health_check.port = None
    else:
        x_health_check.portSpecification = enum_class.USE_FIXED_PORT