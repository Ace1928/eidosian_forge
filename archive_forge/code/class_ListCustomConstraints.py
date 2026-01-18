from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.org_policies import arguments
from googlecloudsdk.command_lib.org_policies import utils
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListCustomConstraints(base.ListCommand):
    """Lists the custom constraints set on an organization."""

    @staticmethod
    def Args(parser):
        arguments.AddOrganizationResourceFlagsToParser(parser)
        parser.display_info.AddFormat("\n        table(\n        name.split('/').slice(-1).join():label=CUSTOM_CONSTRAINT,\n        actionType:label=ACTION_TYPE,\n        method_types.list():label=METHOD_TYPES,\n        resource_types.list():label=RESOURCE_TYPES,\n        display_name:label=DISPLAY_NAME)\n     ")

    def Run(self, args):
        org_policy_client = org_policy_service.OrgPolicyClient(self.ReleaseTrack())
        messages = org_policy_service.OrgPolicyMessages(self.ReleaseTrack())
        parent = utils.GetResourceFromArgs(args)
        request = messages.OrgpolicyOrganizationsCustomConstraintsListRequest(parent=parent)
        return list_pager.YieldFromList(org_policy_client.organizations_customConstraints, request, field='customConstraints', limit=args.limit, batch_size_attribute='pageSize', batch_size=args.page_size)