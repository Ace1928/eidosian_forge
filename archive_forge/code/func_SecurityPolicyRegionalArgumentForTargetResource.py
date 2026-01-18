from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def SecurityPolicyRegionalArgumentForTargetResource(resource, required=False):
    return compute_flags.ResourceArgument(resource_name='security policy', name='--security-policy', completer=RegionalSecurityPoliciesCompleter, plural=False, required=required, regional_collection='compute.regionSecurityPolicies', short_help='The security policy that will be set for this {0}. To remove the policy from this {0} set the policy to an empty string.'.format(resource))