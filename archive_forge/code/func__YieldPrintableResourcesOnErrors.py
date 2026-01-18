from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.deployment_manager import dm_api_util
from googlecloudsdk.api_lib.deployment_manager import dm_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.deployment_manager import alpha_flags
def _YieldPrintableResourcesOnErrors(self, args):
    request = self.messages.DeploymentmanagerResourcesListRequest(project=dm_base.GetProject(), deployment=args.deployment)
    paginated_resources = dm_api_util.YieldWithHttpExceptions(list_pager.YieldFromList(self.client.resources, request, field='resources', limit=args.limit, batch_size=args.page_size))
    for resource in paginated_resources:
        yield resource