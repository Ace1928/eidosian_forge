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
def RunDockerContainer(name, port, env_vars, labels):
    """Runs the Docker container (detached mode) with specified port and name.

  If the name already exists, it will be removed.

  Args:
    name: The name of the container to run.
    port: The port for the container to run on.
    env_vars: The container environment variables.
    labels: Docker labels to store flags and environment variables.

  Raises:
    DockerExecutionException: if the exit code of the execution is non-zero.
  """
    if ContainerExists(name):
        RemoveDockerContainer(name)
    docker_cmd = [_DOCKER, 'run', '-d']
    docker_cmd.extend(['-p', six.text_type(port) + ':8080'])
    if env_vars:
        _AddEnvVars(docker_cmd, env_vars)
    for k, v in labels.items():
        docker_cmd.extend(['--label', '{}={}'.format(k, json.dumps(v))])
    docker_cmd.extend(['--name', name, name])
    status = execution_utils.Exec(docker_cmd, no_exit=True)
    if status:
        raise DockerExecutionException(status, 'Docker failed to run container ' + name)