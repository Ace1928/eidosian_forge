from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddGrpcRelatedUpdateArgs(parser):
    """Adds parser arguments for update subcommands related to gRPC."""
    _AddPortRelatedUpdateArgs(parser, use_port_name=False)
    parser.add_argument('--grpc-service-name', help='      An optional gRPC service name string of up to 1024 characters to include\n      in the gRPC health check request. Pass in an empty string to unset.\n      Only ASCII characters are allowed.')