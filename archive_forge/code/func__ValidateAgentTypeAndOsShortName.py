from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateAgentTypeAndOsShortName(os_short_name, agent_type):
    """Validates the combination of the OS short name and agent type is supported.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field OS short name has been already validated at the arg
  parsing stage. Also the
  other field is OpsAgentPolicy object's 'type' field already complies with the
  allowed values.

  Args:
    os_short_name: str. The OS short name to filter instances by.
    agent_type: str. The AgentRule type.

  Returns:
    An empty list if the validation passes. A singleton list with the following
    error if the validation fails.
    * OSTypeNotSupportedByAgentTypeError:
      The combination of the OS short name and agent type is not supported.
  """
    supported_os_list = _SUPPORTED_OS_SHORT_NAMES_AND_AGENT_TYPES.get(agent_type)
    if os_short_name not in supported_os_list:
        return [OSTypeNotSupportedByAgentTypeError(os_short_name, agent_type)]
    return []