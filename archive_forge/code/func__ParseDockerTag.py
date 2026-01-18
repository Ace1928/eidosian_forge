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
def _ParseDockerTag(tag):
    """Validates and parses a tag string.

  Args:
    tag: str, User input Docker tag string.

  Raises:
    ar_exceptions.InvalidInputValueError if user input is invalid.
    ar_exceptions.UnsupportedLocationError if provided location is invalid.

  Returns:
    A DockerImage and a DockerTag.
  """
    try:
        docker_repo = _ParseInput(tag)
    except ar_exceptions.InvalidInputValueError:
        raise ar_exceptions.InvalidInputValueError(_INVALID_DOCKER_TAG_ERROR)
    img_by_tag_match = re.match(DOCKER_IMG_BY_TAG_REGEX, tag)
    if img_by_tag_match:
        docker_img = DockerImage(docker_repo, img_by_tag_match.group('img'))
        return (docker_img, DockerTag(docker_img, img_by_tag_match.group('tag')))
    else:
        raise ar_exceptions.InvalidInputValueError(_INVALID_DOCKER_TAG_ERROR)