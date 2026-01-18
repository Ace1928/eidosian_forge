from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.org_security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.org_security_policies import flags
from googlecloudsdk.command_lib.compute.org_security_policies import org_security_policies_utils
import six
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class CopyRules(base.UpdateCommand):
    """Replace the rules of a Compute Engine organization security policy with rules from another policy.

  *{command}* is used to replace the rules of organization security policies. An
   organization security policy is a set of rules that controls access to
   various resources.
  """
    ORG_SECURITY_POLICY_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.ORG_SECURITY_POLICY_ARG = flags.OrgSecurityPolicyArgument(required=True, operation='copy the rules to')
        cls.ORG_SECURITY_POLICY_ARG.AddArgument(parser, operation_type='copy-rules')
        flags.AddArgsCopyRules(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        ref = self.ORG_SECURITY_POLICY_ARG.ResolveAsResource(args, holder.resources, with_project=False)
        org_security_policy = client.OrgSecurityPolicy(ref=ref, compute_client=holder.client, resources=holder.resources, version=six.text_type(self.ReleaseTrack()).lower())
        dest_sp_id = org_security_policies_utils.GetSecurityPolicyId(org_security_policy, ref.Name(), organization=args.organization)
        return org_security_policy.CopyRules(only_generate_request=False, dest_sp_id=dest_sp_id, source_security_policy=args.source_security_policy)