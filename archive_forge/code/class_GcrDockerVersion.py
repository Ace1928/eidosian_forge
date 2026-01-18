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
class GcrDockerVersion:
    """Class for sending a gcr.io docker url to container analysis.

  Attributes:
    project:
    docker_string:
  """

    @property
    def project(self):
        return self._project

    def __init__(self, project, docker_string):
        self._project = project
        self._docker_string = docker_string

    def GetDockerString(self):
        return self._docker_string