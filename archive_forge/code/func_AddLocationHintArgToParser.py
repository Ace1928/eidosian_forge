from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddLocationHintArgToParser(parser):
    """Add --location-hint flag."""
    parser.add_argument('--location-hint', hidden=True, help='Used by internal tools to control sub-zone location of node groups.')