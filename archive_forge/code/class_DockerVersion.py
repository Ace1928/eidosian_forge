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
class DockerVersion(object):
    """Holder for a Docker version.

  A valid Docker version has the format of
  LOCATION-docker.DOMAIN/PROJECT-ID/REPOSITORY-ID/IMAGE@sha256:digest

  Properties:
    image: DockerImage, The DockerImage containing the tag.
    digest: str, The name of the Docker digest.
    project: str, the project this image belongs to.
  """

    def __init__(self, docker_img, digest):
        self._image = docker_img
        self._digest = digest

    @property
    def image(self):
        return self._image

    @property
    def digest(self):
        return self._digest

    @property
    def project(self):
        return self._image.docker_repo.project

    def __eq__(self, other):
        if isinstance(other, DockerVersion):
            return self._image == other._image and self._digest == other._digest
        return NotImplemented

    def GetVersionName(self):
        return '{}/versions/{}'.format(self.image.GetPackageName(), self.digest)

    def GetPackageName(self):
        return self.image.GetPackageName()

    def GetDockerString(self):
        return '{}@{}'.format(self.image.GetDockerString(), self.digest)