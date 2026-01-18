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
def GetDockerImage(image_url):
    """Gets a Docker image.

  Args:
    image_url (str): path to a Docker image.

  Returns:
    package: Docker image package

  Throws:
    HttpNotFoundError: if repo or image path are invalid
  """
    image, _ = _ParseDockerImage(image_url, _INVALID_IMAGE_ERROR)
    _ValidateDockerRepo(image.docker_repo.GetRepositoryName())
    return ar_requests.GetPackage(image.GetPackageName())