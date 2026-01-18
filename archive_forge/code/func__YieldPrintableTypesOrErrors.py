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
def _YieldPrintableTypesOrErrors(self, type_providers):
    """Yield dicts of types list, provider, and (optionally) an error message.

    Args:
      type_providers: A dict of project to Type Provider names to grab Type
        Info messages for.

    Yields:
      A dict object with a list of types, a type provider reference (includes
      project) like you would use in Deployment Manager, and (optionally) an
      error message for display.

    """
    for project in type_providers.keys():
        for type_provider in type_providers[project]:
            request = self.messages.DeploymentmanagerTypeProvidersListTypesRequest(project=project, typeProvider=type_provider)
            try:
                paginated_types = dm_api_util.YieldWithHttpExceptions(list_pager.YieldFromList(TypeProviderClient(self.version), request, method='ListTypes', field='types', batch_size=self.page_size, limit=self.limit))
                types = list(paginated_types)
                if types:
                    yield {'types': types, 'provider': project + '/' + type_provider}
            except api_exceptions.HttpException as error:
                self.exit_code = 1
                yield {'types': [], 'provider': project + '/' + type_provider, 'error': error.message}