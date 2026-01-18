from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddAddressFlag(parser):
    """Adds an address flag for service-directory commands."""
    return base.Argument('--address', help='        IPv4 or IPv6 address of the endpoint. The default is\n        empty string.').AddToParser(parser)