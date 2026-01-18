from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
class ForwardingRulesGlobalCompleter(compute_completers.GlobalListCommandCompleter):

    def __init__(self, **kwargs):
        super(ForwardingRulesGlobalCompleter, self).__init__(collection='compute.globalForwardingRules', list_command='compute forwarding-rules list --global --uri', **kwargs)