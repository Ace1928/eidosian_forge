from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddServicesIPV4CIDR(parser):
    parser.add_argument('--services-ipv4-cidr', default='10.96.0.0/12', help='\n      All services in the cluster are assigned an RFC1918 IPv4 address from\n      this block. This field cannot be changed after creation.\n      ')