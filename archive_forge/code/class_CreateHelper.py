from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policies_flags
from googlecloudsdk.command_lib.compute.security_policies import security_policies_utils
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class CreateHelper(object):
    """Create a Compute Engine security policy rule.

  *{command}* is used to create security policy rules.

  ## EXAMPLES

  To create a rule at priority 1000 to block the IP range
  1.2.3.0/24, run:

    $ {command} 1000 \\
       --action=deny-403 \\
       --security-policy=my-policy \\
       --description="block 1.2.3.0/24" \\
       --src-ip-ranges=1.2.3.0/24
  """

    @classmethod
    def Args(cls, parser, support_redirect, support_rate_limit, support_header_action, support_fairshare, support_multiple_rate_limit_keys, support_recaptcha_options):
        """Generates the flagset for a Create command."""
        cls.NAME_ARG = flags.PriorityArgument('add')
        cls.NAME_ARG.AddArgument(parser, operation_type='add', cust_metavar='PRIORITY')
        flags.AddRegionFlag(parser, 'add')
        cls.SECURITY_POLICY_ARG = security_policies_flags.SecurityPolicyMultiScopeArgumentForRules()
        cls.SECURITY_POLICY_ARG.AddArgument(parser)
        flags.AddMatcherAndNetworkMatcher(parser)
        flags.AddAction(parser, support_redirect=support_redirect, support_rate_limit=support_rate_limit, support_fairshare=support_fairshare)
        flags.AddDescription(parser)
        flags.AddPreview(parser, default=None)
        if support_redirect:
            flags.AddRedirectOptions(parser)
        if support_rate_limit:
            flags.AddRateLimitOptions(parser, support_exceed_redirect=support_redirect, support_fairshare=support_fairshare, support_multiple_rate_limit_keys=support_multiple_rate_limit_keys)
        if support_header_action:
            flags.AddRequestHeadersToAdd(parser)
        if support_recaptcha_options:
            flags.AddRecaptchaOptions(parser)
        parser.display_info.AddCacheUpdater(security_policies_flags.SecurityPoliciesCompleter)

    @classmethod
    def Run(cls, release_track, args, support_redirect, support_rate_limit, support_header_action, support_fairshare, support_multiple_rate_limit_keys, support_recaptcha_options):
        """Validates arguments and creates a security policy rule."""
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
        redirect_options = None
        rate_limit_options = None
        if support_redirect:
            redirect_options = security_policies_utils.CreateRedirectOptions(holder.client, args)
        if support_rate_limit:
            rate_limit_options = security_policies_utils.CreateRateLimitOptions(holder.client, args, support_fairshare, support_multiple_rate_limit_keys)
        request_headers_to_add = None
        if support_header_action:
            request_headers_to_add = args.request_headers_to_add
        expression_options = None
        if support_recaptcha_options:
            expression_options = security_policies_utils.CreateExpressionOptions(holder.client, args)
        network_matcher = security_policies_utils.CreateNetworkMatcher(holder.client, args)[0]
        return security_policy_rule.Create(src_ip_ranges=args.src_ip_ranges, expression=args.expression, expression_options=expression_options, network_matcher=network_matcher, action=args.action, description=args.description, preview=args.preview, redirect_options=redirect_options, rate_limit_options=rate_limit_options, request_headers_to_add=request_headers_to_add)