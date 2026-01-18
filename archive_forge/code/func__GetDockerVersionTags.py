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
def _GetDockerVersionTags(client, messages, docker_version):
    """Gets a list of DockerTag associated with the given DockerVersion."""
    tags = ar_requests.ListVersionTags(client, messages, docker_version.GetPackageName(), docker_version.GetVersionName())
    return [DockerTag(docker_version.image, tag.name.split('/')[-1]) for tag in tags]