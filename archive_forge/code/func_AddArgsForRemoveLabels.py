from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddArgsForRemoveLabels(parser):
    """Adds the --labels and --all flags for remove-labels command."""
    args_group = parser.add_mutually_exclusive_group(required=True)
    args_group.add_argument('--all', action='store_true', default=False, help='Remove all labels from the resource.')
    args_group.add_argument('--labels', type=arg_parsers.ArgList(min_length=1), help='\n          A comma-separated list of label keys to remove from the resource.\n          ', metavar='KEY')