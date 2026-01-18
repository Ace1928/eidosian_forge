from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateAgentRulesAndOsTypes(agent_rules, os_types):
    """Validates semantics of the ops-agents-policy.os-types field and the ops-agents-policy.agent-rules field.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is a list of OpsAgentPolicy.Assignment.OsType objects.
  The other field is a list of OpsAgentPolicy.AgentRule object. Each
  OpsAgentPolicy object's 'type' field already complies with the allowed values.

  Args:
    agent_rules: list of OpsAgentPolicy.AgentRule. The list of agent rules to be
      managed by the Ops Agents policy.
    os_types: list of OpsAgentPolicy.Assignment.OsType. The list of OS types as
      part of the instance filters that the Ops Agent policy applies to the Ops
      Agents policy.

  Returns:
    An empty list if the validation passes. A list of errors from the following
    list if the validation fails.
    * OSTypeNotSupportedByAgentTypeError:
      The combination of the OS short name and agent type is not supported.
  """
    errors = []
    for os_type in os_types:
        for agent_rule in agent_rules:
            errors.extend(_ValidateAgentTypeAndOsShortName(os_type.short_name, agent_rule.type))
    return errors