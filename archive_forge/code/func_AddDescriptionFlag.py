from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import arg_parsers
def AddDescriptionFlag(parser):
    """Adds a --description flag to the given parser."""
    help_text = 'Description for the subnet.'
    parser.add_argument('--description', help=help_text, required=False)