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
class DockerTag(object):
    """Holder for a Docker tag.

  A valid Docker tag has the format of
  LOCATION-docker.DOMAIN/PROJECT-ID/REPOSITORY-ID/IMAGE:tag

  Properties:
    image: DockerImage, The DockerImage containing the tag.
    tag: str, The name of the Docker tag.
  """

    def __init__(self, docker_img, tag_id):
        self._image = docker_img
        self._tag = tag_id

    @property
    def image(self):
        return self._image

    @property
    def tag(self):
        return self._tag

    def __eq__(self, other):
        if isinstance(other, DockerTag):
            return self._image == other._image and self._tag == other._tag
        return NotImplemented

    def GetTagName(self):
        return '{}/tags/{}'.format(self.image.GetPackageName(), self.tag)

    def GetPackageName(self):
        return self.image.GetPackageName()

    def GetDockerString(self):
        return '{}:{}'.format(self.image.GetDockerString(), self.tag)