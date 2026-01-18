import os
import pathlib
import string
from typing import Optional
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.instances.ops_agents import cloud_ops_agents_policy as agent_policy
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages as osconfig
def _CreateRollout() -> osconfig.OSPolicyAssignmentRollout:
    return osconfig.OSPolicyAssignmentRollout(disruptionBudget=osconfig.FixedOrPercent(percent=100), minWaitDuration='0s')