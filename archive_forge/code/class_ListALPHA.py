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
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
@dm_base.UseDmApi(dm_base.DmApiVersion.V2BETA)
class ListALPHA(base.ListCommand, dm_base.DmCommand):
    """Describe a type provider type.

  By default, you will see types from your project and gcp-types. To see types
  from any single project, you can use the --provider-project flag.

  ## EXAMPLES

  To print out a list of all available type names, run:

    $ {command}

  If you only want the types for a specific provider, you can specify
  which one using --provider

    $ {command} --provider=PROVIDER

  By default, we'll show you types from your project and gcp-types,
  which contains the default Google Cloud Platform types.
  If you want types for only one project, use the 'provider-project'
  flag. Specifying the provider without a provider-project will search
  both your project and gcp-types for that provider's types.
  """

    @staticmethod
    def Args(parser):
        base.URI_FLAG.RemoveFromParser(parser)
        parser.add_argument('--provider', help='Type provider name.')
        parser.add_argument('--provider-project', help='Project id with types you want to see.')
        parser.display_info.AddFormat('yaml(provider:sort=1, error, types.map().format("{0}", name))')

    def Run(self, args):
        """Run 'types list'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
          with.

    Returns:
      The list of types for this project.

    Raises:
      HttpException: An http error response was received while executing api
          request.
    """
        type_provider_ref = self.resources.Parse(args.provider if args.provider else 'NOT_A_PROVIDER', params={'project': properties.VALUES.core.project.GetOrFail}, collection='deploymentmanager.typeProviders')
        self.page_size = args.page_size
        self.limit = args.limit
        if args.provider_project:
            projects = [args.provider_project]
        else:
            projects = [type_provider_ref.project, GCP_TYPES_PROJECT]
        type_providers = collections.OrderedDict()
        if not args.provider:
            self._GetTypeProviders(projects, type_providers)
        else:
            for project in projects:
                type_providers[project] = [type_provider_ref.typeProvider]
        return dm_api_util.YieldWithHttpExceptions(self._YieldPrintableTypesOrErrors(type_providers))

    def _GetTypeProviders(self, projects, type_providers):
        for project in projects:
            request = self.messages.DeploymentmanagerTypeProvidersListRequest(project=project)
            project_providers = dm_api_util.YieldWithHttpExceptions(list_pager.YieldFromList(TypeProviderClient(self.version), request, field='typeProviders', batch_size=self.page_size, limit=self.limit))
            type_providers[project] = [provider.name for provider in project_providers]

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