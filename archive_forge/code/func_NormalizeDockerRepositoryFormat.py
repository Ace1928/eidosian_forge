from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.functions.v1 import exceptions
from googlecloudsdk.calliope import exceptions as base_exceptions
from six.moves import http_client
def NormalizeDockerRepositoryFormat(docker_repository):
    """Normalizes the docker repository name to the standard resource format.

  Args:
    docker_repository: Fully qualified Docker repository name.

  Returns:
    The name in a standard format supported by the API.
  """
    if docker_repository is None:
        return docker_repository
    repo_match_docker_format = _DOCKER_REPOSITORY_DOCKER_FORMAT_RE.search(docker_repository)
    if repo_match_docker_format:
        project = repo_match_docker_format.group('project')
        location = repo_match_docker_format.group('location')
        name = repo_match_docker_format.group('repo')
        return 'projects/{}/locations/{}/repositories/{}'.format(project, location, name)
    return docker_repository