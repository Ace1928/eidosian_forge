from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import exceptions
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import org_policies_base
from googlecloudsdk.command_lib.resource_manager import org_policies_flags as flags
import six
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Deny(base.Command):
    """Add values to an Organization Policy denied_values list policy.

  Adds one or more values to the specified Organization Policy denied_values
  list policy associated with the specified resource.

  ## EXAMPLES

  The following command adds `devEnv` and `prodEnv` to an Organization Policy
  denied_values list policy for constraint `serviceuser.services`
  on project `foo-project`:

    $ {command} serviceuser.services --project=foo-project devEnv prodEnv
  """

    @staticmethod
    def Args(parser):
        flags.AddIdArgToParser(parser)
        flags.AddParentResourceFlagsToParser(parser)
        base.Argument('denied_value', metavar='DENIED_VALUE', nargs='+', help='The values to add to the denied_values list policy.').AddToParser(parser)

    def Run(self, args):
        messages = org_policies.OrgPoliciesMessages()
        service = org_policies_base.OrgPoliciesService(args)
        policy = service.GetOrgPolicy(org_policies_base.GetOrgPolicyRequest(args))
        if policy.booleanPolicy or (policy.listPolicy and policy.listPolicy.allowedValues):
            raise exceptions.ResourceManagerError('Cannot add values to a non-denied_values list policy.')
        if policy.listPolicy and policy.listPolicy.allValues:
            raise exceptions.ResourceManagerError('Cannot add values if all_values is already specified.')
        if policy.listPolicy and policy.listPolicy.deniedValues:
            for value in args.denied_value:
                policy.listPolicy.deniedValues.append(six.text_type(value))
        else:
            policy.listPolicy = messages.ListPolicy(deniedValues=args.denied_value)
        if policy.restoreDefault:
            policy.restoreDefault = None
        return service.SetOrgPolicy(org_policies_base.SetOrgPolicyRequest(args, policy))