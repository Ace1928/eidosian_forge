from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddInterconnectTypeBetaAndAlpha(parser):
    """Adds interconnect-type flag to the argparse.ArgumentParser."""
    parser.add_argument('--interconnect-type', choices=_INTERCONNECT_TYPE_CHOICES_BETA_AND_ALPHA, action=calliope_actions.DeprecationAction('interconnect-type', removed=False, show_add_help=False, show_message=_ShouldShowDeprecatedWarning, warn='IT_PRIVATE will be deprecated for {flag_name}. Please use DEDICATED instead.', error='Value IT_PRIVATE for {flag_name} has been removed. Please use DEDICATED instead.'), required=True, help='      Type of the interconnect.\n      ')