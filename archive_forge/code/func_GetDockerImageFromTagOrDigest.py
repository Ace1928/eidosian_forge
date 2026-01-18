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
def GetDockerImageFromTagOrDigest(image_name):
    """Gets an image object given either a tag or a digest.

  Args:
    image_name: Either a fully qualified tag or a fully qualified digest.
      Defaults to latest if no tag specified.

  Returns:
    Either a docker_name.Tag or a docker_name.Digest object.

  Raises:
    InvalidImageNameError: Given digest could not be resolved to a full digest.
  """
    if not IsFullySpecified(image_name):
        image_name += ':latest'
    try:
        return ValidateImagePathAndReturn(docker_name.Tag(image_name))
    except docker_name.BadNameException:
        pass
    parts = image_name.split('@', 1)
    if len(parts) == 2:
        if not parts[1].startswith('sha256:'):
            raise InvalidImageNameError('[{0}] digest must be of the form "sha256:<digest>".'.format(image_name))
        if len(parts[1]) < 7 + 64:
            resolved = GetDockerDigestFromPrefix(image_name)
            if resolved == image_name:
                raise InvalidImageNameError('[{0}] could not be resolved to a full digest.'.format(image_name))
            image_name = resolved
    try:
        return ValidateImagePathAndReturn(docker_name.Digest(image_name))
    except docker_name.BadNameException:
        raise InvalidImageNameError('[{0}] digest must be of the form "sha256:<digest>".'.format(image_name))