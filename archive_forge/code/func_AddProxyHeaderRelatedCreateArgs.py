from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddProxyHeaderRelatedCreateArgs(parser, default='NONE'):
    """Adds parser arguments for creation related to ProxyHeader."""
    parser.add_argument('--proxy-header', choices={'NONE': 'No proxy header is added.', 'PROXY_V1': 'Adds the header "PROXY UNKNOWN\\r\\n".'}, default=default, help='The type of proxy protocol header to be sent to the backend.')