from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddCreateAlphaBetaArgs(parser):
    """Adds alpha / beta flags for create command to the argparse.ArgumentParser."""
    AddCreateCommonArgs(parser)
    AddInterconnectTypeBetaAndAlpha(parser)