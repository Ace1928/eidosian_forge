from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
from six.moves import urllib
def GetDockerVersion():
    """Returns the installed Docker client version.

  Returns:
    The installed Docker client version.

  Raises:
    DockerError: Docker cannot be run or does not accept 'docker version
    --format '{{.Client.Version}}''.
  """
    docker_args = "version --format '{{.Client.Version}}'".split()
    docker_p = GetDockerProcess(docker_args, stdin_file=sys.stdin, stdout_file=subprocess.PIPE, stderr_file=subprocess.PIPE)
    stdoutdata, _ = docker_p.communicate()
    if docker_p.returncode != 0 or not stdoutdata:
        raise DockerError('could not retrieve Docker client version')
    return semver.LooseVersion(stdoutdata.strip("'"))