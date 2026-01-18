from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class DescribeHelper(object):
    """Describe a Compute Engine security policy rule.

  *{command}* displays all data associated with a security policy rule.

  ## EXAMPLES

  To describe the rule at priority 1000, run:

    $ {command} 1000 \\
       --security-policy=my-policy
  """
    SECURITY_POLICY_ARG = None
    NAME_ARG = None

    @classmethod
    def Args(cls, parser):
        """Generates the flagset for a Describe command."""
        cls.NAME_ARG = flags.PriorityArgument('describe')
        cls.NAME_ARG.AddArgument(parser, operation_type='describe', cust_metavar='PRIORITY')
        flags.AddRegionFlag(parser, 'describe')
        cls.SECURITY_POLICY_ARG = security_policy_flags.SecurityPolicyMultiScopeArgumentForRules()
        cls.SECURITY_POLICY_ARG.AddArgument(parser)

    @classmethod
    def Run(cls, release_track, args):
        """Validates arguments and describes a security policy rule."""
        holder = base_classes.ComputeApiHolder(release_track)
        if args.security_policy:
            security_policy_ref = cls.SECURITY_POLICY_ARG.ResolveAsResource(args, holder.resources, default_scope=compute_scope.ScopeEnum.GLOBAL)
            if getattr(security_policy_ref, 'region', None) is not None:
                ref = holder.resources.Parse(args.name, collection='compute.regionSecurityPolicyRules', params={'project': properties.VALUES.core.project.GetOrFail, 'region': security_policy_ref.region, 'securityPolicy': args.security_policy})
            else:
                ref = holder.resources.Parse(args.name, collection='compute.securityPolicyRules', params={'project': properties.VALUES.core.project.GetOrFail, 'securityPolicy': args.security_policy})
        else:
            try:
                ref = holder.resources.Parse(args.name, collection='compute.regionSecurityPolicyRules', params={'project': properties.VALUES.core.project.GetOrFail, 'region': getattr(args, 'region', None)})
            except (resources.RequiredFieldOmittedException, resources.WrongResourceCollectionException):
                ref = holder.resources.Parse(args.name, collection='compute.securityPolicyRules', params={'project': properties.VALUES.core.project.GetOrFail})
        security_policy_rule = client.SecurityPolicyRule(ref, compute_client=holder.client)
        return security_policy_rule.Describe()