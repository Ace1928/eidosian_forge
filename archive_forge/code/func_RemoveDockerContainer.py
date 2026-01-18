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
def RemoveDockerContainer(name):
    """Removes the Docker container with specified name.

  Args:
    name: The name of the Docker container to delete.

  Raises:
    DockerExecutionException: if the exit code of the execution is non-zero.
  """
    delete_cmd = [_DOCKER, 'rm', '-f', name]
    status = execution_utils.Exec(delete_cmd, no_exit=True)
    if status:
        raise DockerExecutionException(status, 'Docker failed to execute: failed to remove container ' + name)