from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddLinkType(parser):
    """Adds link-type flag to the argparse.ArgumentParser."""
    link_types = _LINK_TYPE_CHOICES
    parser.add_argument('--link-type', choices=link_types, required=True, help='      Type of the link for the interconnect.\n      ')