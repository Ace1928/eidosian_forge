from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddPreview(parser, default):
    """Adds the preview argument to the argparse."""
    parser.add_argument('--preview', action='store_true', default=default, help='If specified, the action will not be enforced.')