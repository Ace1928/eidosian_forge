from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def AddCapacityLimits(parser, support_global_neg=False, support_region_neg=False):
    """Adds capacity thresholds arguments to the argparse."""
    AddMaxUtilization(parser)
    capacity_group = parser.add_group(mutex=True)
    capacity_incompatible_types = []
    if support_global_neg:
        capacity_incompatible_types.extend(['INTERNET_IP_PORT', 'INTERNET_FQDN_PORT'])
    if support_region_neg:
        capacity_incompatible_types.append('SERVERLESS')
    append_help_text = '\n  This cannot be used when the endpoint type of an attached network endpoint\n  group is {0}.\n  '.format(_JoinTypes(capacity_incompatible_types)) if capacity_incompatible_types else ''
    capacity_group.add_argument('--max-rate-per-endpoint', type=float, help='      Only valid for network endpoint group backends. Defines a maximum\n      number of HTTP requests per second (RPS) per endpoint if all endpoints\n      are healthy. When one or more endpoints are unhealthy, an effective\n      maximum rate per healthy endpoint is calculated by multiplying\n      `MAX_RATE_PER_ENDPOINT` by the number of endpoints in the network\n      endpoint group, and then dividing by the number of healthy endpoints.\n      ' + append_help_text)
    capacity_group.add_argument('--max-connections-per-endpoint', type=int, help='      Only valid for network endpoint group backends. Defines a maximum\n      number of connections per endpoint if all endpoints are healthy. When\n      one or more endpoints are unhealthy, an effective maximum average number\n      of connections per healthy endpoint is calculated by multiplying\n      `MAX_CONNECTIONS_PER_ENDPOINT` by the number of endpoints in the network\n      endpoint group, and then dividing by the number of healthy endpoints.\n      ' + append_help_text)
    capacity_group.add_argument('--max-rate', type=int, help='      Maximum number of HTTP requests per second (RPS) that the backend can\n      handle. Valid for network endpoint group and instance group backends\n      (except for regional managed instance groups). Must not be defined if the\n      backend is a managed instance group using load balancing-based autoscaling.\n      ' + append_help_text)
    capacity_group.add_argument('--max-rate-per-instance', type=float, help='      Only valid for instance group backends. Defines a maximum number of\n      HTTP requests per second (RPS) per instance if all instances in the\n      instance group are healthy. When one or more instances are unhealthy,\n      an effective maximum RPS per healthy instance is calculated by\n      multiplying `MAX_RATE_PER_INSTANCE` by the number of instances in the\n      instance group, and then dividing by the number of healthy instances. This\n      parameter is compatible with managed instance group backends that use\n      autoscaling based on load balancing.\n      ')
    capacity_group.add_argument('--max-connections', type=int, help='      Maximum concurrent connections that the backend can handle. Valid for\n      network endpoint group and instance group backends (except for regional\n      managed instance groups).\n      ' + append_help_text)
    capacity_group.add_argument('--max-connections-per-instance', type=int, help='      Only valid for instance group backends. Defines a maximum number\n      of concurrent connections per instance if all instances in the\n      instance group are healthy. When one or more instances are\n      unhealthy, an effective average maximum number of connections per healthy\n      instance is calculated by multiplying `MAX_CONNECTIONS_PER_INSTANCE`\n      by the number of instances in the instance group, and then dividing by\n      the number of healthy instances.\n      ')