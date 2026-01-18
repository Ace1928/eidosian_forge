import dataclasses
import enum
import json
import sys
from typing import Any, Mapping, Optional
from apitools.base.py import encoding
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages
def CreateOpsAgentsPolicy(ops_agents_policy: Mapping[str, Any]) -> OpsAgentsPolicy:
    """Create Ops Agent Policy.

  Args:
    ops_agents_policy: fields (agents_rule, instance_filter) describing ops
      agents policy from the command line.

  Returns:
    Ops agents policy.
  """
    if not ops_agents_policy or ops_agents_policy.keys() != _OPS_AGENTS_POLICY_KEYS:
        raise ValueError('ops_agents_policy must contain agents_rule and instance_filter')
    return OpsAgentsPolicy(agents_rule=CreateAgentsRule(ops_agents_policy['agents_rule']), instance_filter=encoding.PyValueToMessage(osconfig_v1_messages.OSPolicyAssignmentInstanceFilter, ops_agents_policy['instance_filter']))