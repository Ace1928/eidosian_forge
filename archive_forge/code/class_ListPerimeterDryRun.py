from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ListPerimeterDryRun(base.ListCommand):
    """Lists the effective dry-run configuration across all Service Perimeters."""
    _API_VERSION = 'v1'

    @staticmethod
    def Args(parser):
        base.URI_FLAG.RemoveFromParser(parser)
        parser.add_argument('--policy', metavar='policy', default=None, help='Policy resource - The access policy you want to list the\n                effective dry-run configuration for. This represents a Cloud\n                resource.')
        parser.display_info.AddFormat('yaml(name.basename(), title, spec)')

    def Run(self, args):
        client = zones_api.Client(version=self._API_VERSION)
        policy_id = policies.GetDefaultPolicy()
        if args.IsSpecified('policy'):
            policy_id = args.policy
        policy_ref = resources.REGISTRY.Parse(policy_id, collection='accesscontextmanager.accessPolicies')
        perimeters_to_display = [p for p in client.List(policy_ref)]
        for p in perimeters_to_display:
            if not p.useExplicitDryRunSpec:
                p.spec = p.status
                p.name += '*'
            p.status = None
        print("Note: Perimeters marked with '*' do not have an explicit `spec`. Instead, their `status` also acts as the `spec`.")
        return perimeters_to_display