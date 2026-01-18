from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddClusterIPV6CIDR(parser):
    parser.add_argument('--cluster-ipv6-cidr', help='\n      If specified, all pods in the cluster are assigned an RFC4193 IPv6 address\n      from this block. This field cannot be changed after creation.\n      ')