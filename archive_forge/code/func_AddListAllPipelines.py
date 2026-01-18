from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddListAllPipelines(parser):
    """Add --list-all-pipelines flag."""
    help_text = textwrap.dedent('  List all Delivery Pipelines associated with a target.\n\n  Usage:\n\n    $ {command} --list-all-pipelines\n\n')
    parser.add_argument('--list-all-pipelines', action='store_true', default=None, help=help_text)