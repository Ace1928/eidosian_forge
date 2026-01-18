from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def InterconnectArgumentForOtherResource(short_help, required=True, detailed_help=None):
    return compute_flags.ResourceArgument(name='--interconnect', resource_name='interconnect', completer=InterconnectsCompleter, plural=False, required=required, global_collection='compute.interconnects', short_help=short_help, detailed_help=detailed_help)