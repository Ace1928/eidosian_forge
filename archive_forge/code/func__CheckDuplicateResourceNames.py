from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _CheckDuplicateResourceNames(resource_type_to_names):
    """Checks if there are any duplicate resource names per resource type.

  Args:
     resource_type_to_names: dict[str,[str]], dict of resource type and names.

  Raises:
    exceptions.CloudDeployConfigError, if there are duplicate names for a given
    resource type.
  """
    errors = []
    for k, names in resource_type_to_names.items():
        dups = set()
        for name in names:
            if names.count(name) > 1:
                dups.add(name)
        if dups:
            errors.append('{} has duplicate name(s): {}'.format(k, dups))
    if errors:
        raise exceptions.CloudDeployConfigError(errors)