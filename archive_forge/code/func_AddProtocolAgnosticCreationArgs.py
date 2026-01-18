from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddProtocolAgnosticCreationArgs(parser, protocol_string):
    """Adds parser arguments common to creation for all protocols."""
    parser.add_argument('--check-interval', type=arg_parsers.Duration(), default='5s', help="      How often to perform a health check for an instance. For example,\n      specifying ``10s'' will run the check every 10 seconds. The default\n      value is ``5s''.\n      See $ gcloud topic datetimes for information on duration formats.\n       ")
    parser.add_argument('--timeout', type=arg_parsers.Duration(), default='5s', help="      If Google Compute Engine doesn't receive a healthy response from the\n      instance by the time specified by the value of this flag, the health\n      check request is considered a failure. For example, specifying ``10s''\n      will cause the check to wait for 10 seconds before considering the\n      request a failure. The default value is ``5s''.\n      See $ gcloud topic datetimes for information on duration formats.\n      ")
    parser.add_argument('--unhealthy-threshold', type=int, default=2, help='      The number of consecutive health check failures before a healthy\n      instance is marked as unhealthy. The default is 2.\n      ')
    parser.add_argument('--healthy-threshold', type=int, default=2, help='      The number of consecutive successful health checks before an\n      unhealthy instance is marked as healthy. The default is 2.\n      ')
    parser.add_argument('--description', help='      An optional string description for the ' + protocol_string + ' health\n      check.\n      ')