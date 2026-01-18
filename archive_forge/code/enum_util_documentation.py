from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
Converts string value of the docker_registry enum to its enum equivalent.

  Args:
    docker_registry_str: a string representing the enum value

  Returns:
    Corresponding DockerRegistryValueValuesEnum value or None for invalid values
  