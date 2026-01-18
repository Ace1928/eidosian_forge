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
def DockerUrlToVersion(url):
    """Validates a Docker image URL and get Docker version information.

  Args:
    url: Url of a docker image.

  Returns:
    A DockerImage, and a DockerVersion.

  Raises:
    ar_exceptions.InvalidInputValueError: If user input is invalid.

  """
    image, version_or_tag = _ParseDockerImage(url, _INVALID_IMAGE_ERROR, strict=False)
    _ValidateDockerRepo(image.docker_repo.GetRepositoryName())
    docker_version = _ValidateAndGetDockerVersion(version_or_tag)
    return (image, docker_version)