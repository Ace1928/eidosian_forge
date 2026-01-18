from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import textwrap
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core.util import files
import six
def GetDockerContainerLabels(name):
    """Returns the labels of the Docker container with specified name.

  Args:
    name: The name of the Docker container.

  Returns:
    dict: The labels for the docker container in json format.

  Raises:
    DockerExecutionException: if the exit code of the execution is non-zero
    or if the port of the container does not exist.
  """
    if not ContainerExists(name):
        return {}
    find_labels = [_DOCKER, 'inspect', '--format={{json .Config.Labels}}', name]
    out = []
    capture_out = lambda stdout: out.append(stdout.strip())
    status = execution_utils.Exec(find_labels, out_func=capture_out, no_exit=True)
    if status:
        raise DockerExecutionException(status, 'Docker failed to execute: failed to labels for ' + name)
    return json.loads(out[0])