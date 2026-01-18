from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def PriorityArgument(operation, is_plural=False):
    return compute_flags.ResourceArgument('name' + ('s' if is_plural else ''), completer=SecurityPolicyRulesCompleter, global_collection='compute.securityPolicyRules', regional_collection='compute.regionSecurityPolicyRules', region_hidden=False, scope_flags_usage=compute_flags.ScopeFlagsUsage.DONT_USE_SCOPE_FLAGS, plural=is_plural, required=False if is_plural else True, detailed_help='The priority of the rule{0} to {1}. Rules are evaluated in order from highest priority to lowest priority where 0 is the highest priority and 2147483647 is the lowest priority.'.format('s' if is_plural else '', operation))