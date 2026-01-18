from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import firewalls_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute.firewall_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.firewall_policies import firewall_policies_utils
from googlecloudsdk.command_lib.compute.firewall_policies import flags
from googlecloudsdk.core import log
import six
class ListRules(base.DescribeCommand, base.ListCommand):
    """List the rules of a Compute Engine organization firewall policy.

  *{command}* is used to list the rules of an organization firewall policy.
  """
    FIREWALL_POLICY_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.FIREWALL_POLICY_ARG = flags.FirewallPolicyArgument(required=True, operation='list rules for')
        cls.FIREWALL_POLICY_ARG.AddArgument(parser, operation_type='get')
        parser.add_argument('--organization', help='Organization which the organization firewall policy belongs to. Must be set if FIREWALL_POLICY is short name.')
        parser.display_info.AddFormat(DEFAULT_LIST_FORMAT)
        lister.AddBaseListerArgs(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        ref = self.FIREWALL_POLICY_ARG.ResolveAsResource(args, holder.resources, with_project=False)
        org_firewall_policy = client.OrgFirewallPolicy(ref=ref, compute_client=holder.client, resources=holder.resources, version=six.text_type(self.ReleaseTrack()).lower())
        fp_id = firewall_policies_utils.GetFirewallPolicyId(org_firewall_policy, ref.Name(), organization=args.organization)
        response = org_firewall_policy.Describe(fp_id=fp_id, only_generate_request=False)
        if not response:
            return None
        return firewalls_utils.SortFirewallPolicyRules(holder.client, response[0].rules)

    def Epilog(self, resources_were_displayed):
        del resources_were_displayed
        log.status.Print('\n' + LIST_NOTICE)