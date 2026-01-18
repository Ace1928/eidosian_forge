from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from contextlib import contextmanager
import re
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2 import docker_http as v2_docker_http
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list
from googlecloudsdk.api_lib.container.images import container_analysis_data_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.core.util import times
import six
from six.moves import map
import six.moves.http_client
def GetDockerDigestFromPrefix(digest):
    """Gets a full digest string given a potential prefix.

  Args:
    digest: The digest prefix

  Returns:
    The full digest, or the same prefix if no full digest is found.

  Raises:
    InvalidImageNameError: if the prefix supplied isn't unique.
  """
    repository_path, prefix = digest.split('@', 1)
    repository = ValidateRepositoryPath(repository_path)
    with v2_2_image.FromRegistry(basic_creds=CredentialProvider(), name=repository, transport=Http()) as image:
        matches = [d for d in image.manifests() if d.startswith(prefix)]
        if len(matches) == 1:
            return repository_path + '@' + matches.pop()
        elif len(matches) > 1:
            raise InvalidImageNameError('{0} is not a unique digest prefix. Options are {1}.]'.format(prefix, ', '.join(map(str, matches))))
        return digest