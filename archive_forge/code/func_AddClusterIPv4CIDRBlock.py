from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddClusterIPv4CIDRBlock(parser):
    """Adds --cluster-ipv4-cidr-block flag."""
    parser.add_argument('--cluster-ipv4-cidr-block', help="The IP address range for the cluster pod IPs. Can be specified as a netmask size (e.g. '/14') or as in CIDR notation (e.g. '10.100.0.0/20'). Defaults to '/20' if flag is not provided.")