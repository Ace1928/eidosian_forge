from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import policies as policies_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import policies
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdatePoliciesGA(base.UpdateCommand):
    """Update an existing access policy."""
    _API_VERSION = 'v1'

    @staticmethod
    def Args(parser):
        policies.AddResourceArg(parser, 'to update')
        common.GetTitleArg('access policy').AddToParser(parser)

    def Run(self, args):
        client = policies_api.Client(version=self._API_VERSION)
        policy_ref = args.CONCEPTS.policy.Parse()
        return client.Patch(policy_ref, title=args.title)