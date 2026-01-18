from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddMaxEndpointsFlag(parser):
    """Adds max_endpoints flags for service-directory commands."""
    return base.Argument('--max-endpoints', type=int, help='           Maximum number of endpoints to return.\n           ').AddToParser(parser)