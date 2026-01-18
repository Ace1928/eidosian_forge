from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateOnlyOneOsTypeAllowed(os_types):
    """Validates that no more than one OS type is specified.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is a list of OpsAgentPolicy.Assignment.OsType objects.

  Args:
    os_types: list of OpsAgentPolicy.Assignment.OsType.
      The list of OS types as part of the instance filters that the Ops Agent
      policy applies to the Ops Agents policy.

  Returns:
    An empty list if the validation passes. A singleton list with the following
    error if the validation fails.
    * OsTypesMoreThanOneError:
      More than one OS types are specified.
  """
    if len(os_types) > 1:
        return [OsTypesMoreThanOneError()]
    return []