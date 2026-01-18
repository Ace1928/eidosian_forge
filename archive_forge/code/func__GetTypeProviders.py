from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.deployment_manager import dm_api_util
from googlecloudsdk.api_lib.deployment_manager import dm_base
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _GetTypeProviders(self, projects, type_providers):
    for project in projects:
        request = self.messages.DeploymentmanagerTypeProvidersListRequest(project=project)
        project_providers = dm_api_util.YieldWithHttpExceptions(list_pager.YieldFromList(TypeProviderClient(self.version), request, field='typeProviders', batch_size=self.page_size, limit=self.limit))
        type_providers[project] = [provider.name for provider in project_providers]