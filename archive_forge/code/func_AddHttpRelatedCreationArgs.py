from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddHttpRelatedCreationArgs(parser, include_weighted_load_balancing=False):
    """Adds parser arguments for creation related to HTTP."""
    _AddPortRelatedCreationArgs(parser)
    AddProxyHeaderRelatedCreateArgs(parser)
    parser.add_argument('--host', help="      The value of the host header used for the health check. If unspecified,\n      Google Cloud sets the host header to the IP address of the load balancer's\n      forwarding rule.\n      ")
    parser.add_argument('--request-path', default='/', help="      The request path that this health check monitors. For example,\n      ``/healthcheck''. The default value is ``/''.\n      ")
    if include_weighted_load_balancing:
        parser.add_argument('--weight-report-mode', choices=['ENABLE', 'DISABLE', 'DRY_RUN'], help='        Defines whether Weighted Load Balancing is enabled.\n        ')