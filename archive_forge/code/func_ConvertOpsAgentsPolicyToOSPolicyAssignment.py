import os
import pathlib
import string
from typing import Optional
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.instances.ops_agents import cloud_ops_agents_policy as agent_policy
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages as osconfig
def ConvertOpsAgentsPolicyToOSPolicyAssignment(name: str, ops_agents_policy: agent_policy.OpsAgentsPolicy) -> osconfig.OSPolicyAssignment:
    """Converts Ops Agent policy to OS Config guest policy."""
    os_policy = _CreateOSPolicy(agents_rule=ops_agents_policy.agents_rule)
    os_rollout = _CreateRollout()
    return osconfig.OSPolicyAssignment(name=name, osPolicies=[os_policy], instanceFilter=ops_agents_policy.instance_filter, rollout=os_rollout, description='Cloud Ops Policy Assignment via gcloud')