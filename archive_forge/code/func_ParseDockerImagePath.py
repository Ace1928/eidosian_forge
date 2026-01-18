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
def ParseDockerImagePath(img_path):
    """Validates and parses an image path into a DockerImage or a DockerRepo."""
    if not img_path:
        return _GetDefaultResources()
    resource_val_list = list(filter(None, img_path.split('/')))
    try:
        docker_repo = _ParseInput(img_path)
    except ar_exceptions.InvalidInputValueError:
        raise ar_exceptions.InvalidInputValueError(_INVALID_IMAGE_PATH_ERROR)
    if len(resource_val_list) == 3:
        return docker_repo
    elif len(resource_val_list) > 3:
        return DockerImage(docker_repo, '/'.join(resource_val_list[3:]))
    raise ar_exceptions.InvalidInputValueError(_INVALID_IMAGE_PATH_ERROR)