from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddInitialRolloutGroup(parser):
    """Adds initial-rollout flag group."""
    group = parser.add_mutually_exclusive_group()
    enable_initial_rollout_group = group.add_group(mutex=False)
    AddInitialRolloutLabelsFlag().AddToParser(enable_initial_rollout_group)
    AddInitialRolloutAnnotationsFlag().AddToParser(enable_initial_rollout_group)
    AddInitialRolloutPhaseIDFlag().AddToParser(enable_initial_rollout_group)
    AddEnableInitialRolloutFlag().AddToParser(enable_initial_rollout_group)
    AddDisableInitialRolloutFlag().AddToParser(group)