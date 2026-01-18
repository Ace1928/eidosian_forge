from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddNetworkEndpointGroupType(parser, support_neg_type):
    """Adds NEG type argument for creating network endpoint group."""
    if support_neg_type:
        base.ChoiceArgument('--neg-type', hidden=True, choices=['load-balancing'], default='load-balancing', help_str='The type of network endpoint group to create.').AddToParser(parser)