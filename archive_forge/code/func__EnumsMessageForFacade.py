from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.util.apis import arg_utils
def _EnumsMessageForFacade(facade: str):
    """Returns the PublicRepository enum messages for a facade."""
    facade_to_enum = {'Maven': ar_requests.GetMessages().MavenRepository().PublicRepositoryValueValuesEnum, 'Docker': ar_requests.GetMessages().DockerRepository().PublicRepositoryValueValuesEnum, 'Npm': ar_requests.GetMessages().NpmRepository().PublicRepositoryValueValuesEnum, 'Python': ar_requests.GetMessages().PythonRepository().PublicRepositoryValueValuesEnum, 'Apt': ar_requests.GetMessages().GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigAptRepositoryPublicRepository().RepositoryBaseValueValuesEnum, 'Yum': ar_requests.GetMessages().GoogleDevtoolsArtifactregistryV1RemoteRepositoryConfigYumRepositoryPublicRepository().RepositoryBaseValueValuesEnum}
    return facade_to_enum[facade]