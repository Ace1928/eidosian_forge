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
def GetDockerTagsForDigest(digest, http_obj):
    """Gets all of the tags for a given digest.

  Args:
    digest: docker_name.Digest, The digest supplied by a user.
    http_obj: http.Http(), The http transport.

  Returns:
    A list of all of the tags associated with the input digest.
  """
    repository_path = digest.registry + '/' + digest.repository
    repository = ValidateRepositoryPath(repository_path)
    tags = []
    tag_names = GetTagNamesForDigest(digest, http_obj)
    for tag_name in tag_names:
        try:
            tag = docker_name.Tag(six.text_type(repository) + ':' + tag_name)
        except docker_name.BadNameException as e:
            raise InvalidImageNameError(six.text_type(e))
        tags.append(tag)
    return tags