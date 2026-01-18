from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddNetworkTags(parser):
    """Adds a --network-tags flag to the given parser."""
    help_text = '  Network tags to add to the Google Compute Engine machines backing the\n  Workstations.\n\n  Example:\n\n    $ {command} --network-tags=tag_1,tag_2\n  '
    parser.add_argument('--network-tags', metavar='NETWORK_TAGS', type=arg_parsers.ArgList(), help=help_text)