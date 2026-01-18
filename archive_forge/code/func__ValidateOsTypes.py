from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateOsTypes(os_types):
    """Validates semantics of the ops-agents-policy.os-types field.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is a list of OpsAgentPolicy.Assignment.OsType objects.

  Args:
    os_types: list of OpsAgentPolicy.Assignment.OsType.
      The list of OS types as part of the instance filters that the Ops Agent
      policy applies to the Ops Agents policy.

  Returns:
    An empty list if the validation passes. A list of errors from the following
    list if the validation fails.
    * OsTypesMoreThanOneError:
      More than one OS types are specified.
    * OsTypeNotSupportedError:
      The combination of the OS short name and version is not supported.
  """
    errors = _ValidateOnlyOneOsTypeAllowed(os_types)
    for os_type in os_types:
        errors.extend(_ValidateSupportedOsType(os_type.short_name, os_type.version))
    return errors