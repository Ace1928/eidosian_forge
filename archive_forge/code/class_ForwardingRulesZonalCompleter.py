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
class ForwardingRulesZonalCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(ForwardingRulesZonalCompleter, self).__init__(collection='compute.forwardingRules', list_command='compute forwarding-rules list --filter=region:* --uri', **kwargs)