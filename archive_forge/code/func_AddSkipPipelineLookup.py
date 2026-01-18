from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddSkipPipelineLookup(parser):
    """Add --skip-pipeline-lookup flag."""
    help_text = textwrap.dedent('  If set, skip fetching details of associated pipelines when describing a target.\n\n  Usage:\n\n    $ {command} --skip-pipeline-lookup\n\n')
    parser.add_argument('--skip-pipeline-lookup', action='store_true', default=False, help=help_text)