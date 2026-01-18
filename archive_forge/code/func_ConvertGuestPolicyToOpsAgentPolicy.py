from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.calliope import exceptions
def ConvertGuestPolicyToOpsAgentPolicy(guest_policy):
    """Converts OS Config guest policy to Ops Agent policy."""
    description, agent_rules = _ExtractDescriptionAndAgentRules(guest_policy.description)
    return agent_policy.OpsAgentPolicy(assignment=_CreateAssignment(guest_policy.assignment), agent_rules=_CreateAgentRules(agent_rules), description=description, etag=guest_policy.etag, name=guest_policy.name, update_time=guest_policy.updateTime, create_time=guest_policy.createTime)