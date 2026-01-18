from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.images.packages import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddShowAddedPackagesFlag(parser, use_default_value=True):
    """Add --show-added-packages Boolean flag."""
    help_text = 'Show only the packages added to the diff image.'
    action = 'store_true' if use_default_value else arg_parsers.StoreTrueFalseAction
    parser.add_argument('--show-added-packages', help=help_text, action=action)