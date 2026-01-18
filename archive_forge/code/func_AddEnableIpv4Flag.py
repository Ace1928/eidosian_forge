from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddEnableIpv4Flag(parser):
    """Adds a --enable-ip-v4 flag to the given parser."""
    help_text = 'Whether the instance should be assigned an IPv4 address or not.'
    parser.add_argument('--enable-ip-v4', help=help_text, action='store_true', dest='enable_ip_v4', default=True)