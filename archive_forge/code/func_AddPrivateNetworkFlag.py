from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddPrivateNetworkFlag(parser):
    """Adds a --private-network flag to the given parser."""
    help_text = '    Resource link for the VPC network from which the Cloud SQL instance is\n    accessible for private IP. For example,\n    /projects/myProject/global/networks/default. This setting can be updated,\n    but it cannot be removed after it is set.\n    '
    parser.add_argument('--private-network', help=help_text)