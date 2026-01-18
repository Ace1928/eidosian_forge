from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateAgentTypesConflict(agent_rules):
    """Validates that when agent type is ops-agent, it is the only agent type.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is a list of OpsAgentPolicy.AgentRule object. Each
  OpsAgentPolicy object's 'type' field already complies with the allowed values.

  Args:
    agent_rules: list of OpsAgentPolicy.AgentRule. The list of agent rules to be
      managed by the Ops Agents policy.

  Returns:
    An empty list if the validation passes. A list that contains one or more
    errors below if the validation fails.
    * AgentTypesConflictError:
      More than one agent type is specified when there is already a type
      ops-agent.
  """
    agent_types = {agent_rule.type for agent_rule in agent_rules}
    if agent_policy.OpsAgentPolicy.AgentRule.Type.OPS_AGENT in agent_types and len(agent_types) > 1:
        return [AgentTypesConflictError()]
    return []