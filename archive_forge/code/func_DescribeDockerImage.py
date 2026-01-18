from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.artifacts import containeranalysis_util as ca_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def DescribeDockerImage(args):
    """Retrieves information about a docker image based on the fully-qualified name.

  Args:
    args: user input arguments.

  Returns:
    A dictionary of information about the given docker image.
  """
    ar_image_name, gcr_project, in_gcr_format = ConvertGCRImageString(args.IMAGE)
    if in_gcr_format:
        messages = ar_requests.GetMessages()
        settings = ar_requests.GetProjectSettings(gcr_project)
        if settings.legacyRedirectionState != messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_ENABLED:
            raise ar_exceptions.InvalidInputValueError('This command only supports Artifact Registry. You can enable redirection to use gcr.io repositories in Artifact Registry.')
    image, docker_version = DockerUrlToVersion(ar_image_name)
    scanning_allowed = True
    scanning_docker_version = docker_version
    if 'gcr.io' in image.docker_repo.repo:
        if not in_gcr_format:
            messages = ar_requests.GetMessages()
            settings = ar_requests.GetProjectSettings(image.docker_repo.project)
            if settings.legacyRedirectionState != messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_ENABLED:
                log.warning('gcr.io domain repos in Artifact Registry are not scanned unless they are redirected')
                scanning_allowed = False
            else:
                log.info('Note: The container scanning API uses the gcr.io url for gcr.io domain repos')
        scanning_docker_version = GcrDockerVersion(image.docker_repo.project, docker_version.GetDockerString().replace(image.docker_repo.GetDockerString(), '{}/{}'.format(image.docker_repo.repo, image.docker_repo.project)))
    result = {}
    result['image_summary'] = {'digest': docker_version.digest, 'fully_qualified_digest': docker_version.GetDockerString(), 'registry': '{}-docker.{}'.format(docker_version.image.docker_repo.location, properties.VALUES.artifacts.domain.Get()), 'repository': docker_version.image.docker_repo.repo}
    if scanning_allowed:
        summary_metadata = ca_util.GetImageSummaryMetadata(scanning_docker_version)
        result['image_summary']['slsa_build_level'] = summary_metadata.SLSABuildLevel()
        sbom_locations = summary_metadata.SbomLocations()
        if sbom_locations:
            result['image_summary']['sbom_locations'] = sbom_locations
        metadata = ca_util.GetContainerAnalysisMetadata(scanning_docker_version, args)
        result.update(metadata.ArtifactsDescribeView())
    return result