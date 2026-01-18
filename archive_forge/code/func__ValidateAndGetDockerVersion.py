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
def _ValidateAndGetDockerVersion(version_or_tag):
    """Validates a version_or_tag and returns the validated DockerVersion object.

  Args:
    version_or_tag: a docker version or a docker tag.

  Returns:
    a DockerVersion object.

  Raises:
    ar_exceptions.InvalidInputValueError if version_or_tag is not valid.
  """
    try:
        if isinstance(version_or_tag, DockerVersion):
            ar_requests.GetVersion(ar_requests.GetClient(), ar_requests.GetMessages(), version_or_tag.GetVersionName())
            return version_or_tag
        elif isinstance(version_or_tag, DockerTag):
            digest = ar_requests.GetVersionFromTag(ar_requests.GetClient(), ar_requests.GetMessages(), version_or_tag.GetTagName())
            docker_version = DockerVersion(version_or_tag.image, digest)
            return docker_version
        else:
            raise ar_exceptions.InvalidInputValueError(_INVALID_DOCKER_IMAGE_ERROR)
    except api_exceptions.HttpNotFoundError:
        raise ar_exceptions.InvalidInputValueError(_DOCKER_IMAGE_NOT_FOUND)