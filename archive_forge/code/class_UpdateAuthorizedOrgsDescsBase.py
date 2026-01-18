from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import authorized_orgs as authorized_orgs_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import authorized_orgs
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdateAuthorizedOrgsDescsBase(base.UpdateCommand):
    """Update an existing authorized organizations description."""
    _API_VERSION = 'v1'

    @staticmethod
    def Args(parser):
        UpdateAuthorizedOrgsDescsBase.ArgsVersioned(parser)

    @staticmethod
    def ArgsVersioned(parser):
        authorized_orgs.AddResourceArg(parser, 'to update')
        authorized_orgs.AddAuthorizedOrgsDescUpdateArgs(parser)

    def Run(self, args):
        client = authorized_orgs_api.Client(version=self._API_VERSION)
        authorized_orgs_desc_ref = args.CONCEPTS.authorized_orgs_desc.Parse()
        result = repeated.CachedResult.FromFunc(client.Get, authorized_orgs_desc_ref)
        policies.ValidateAccessPolicyArg(authorized_orgs_desc_ref, args)
        return self.Patch(client=client, authorized_orgs_desc_ref=authorized_orgs_desc_ref, orgs=authorized_orgs.ParseOrgs(args, result))

    def Patch(self, client, authorized_orgs_desc_ref, orgs):
        return client.Patch(authorized_orgs_desc_ref, orgs=orgs)