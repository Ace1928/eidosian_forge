from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import sys
from googlecloudsdk.core.resource import resource_property
def CreateOpsAgentPolicy(description, agent_rules, group_labels, os_types, zones, instances):
    """Create Ops Agent Policy.

  Args:
    description: str, ops agent policy description.
    agent_rules: list of dict, fields describing agent rules from the command
      line.
    group_labels: list of dict, VM group label matchers.
    os_types: dict, VM OS type matchers.
    zones: list, VM zone matchers.
    instances: list, instance name matchers.

  Returns:
    ops agent policy.
  """
    return OpsAgentPolicy(assignment=OpsAgentPolicy.Assignment(group_labels=group_labels, zones=zones, instances=instances, os_types=CreateOsTypes(os_types)), agent_rules=CreateAgentRules(agent_rules), description=description, etag=None, name=None, update_time=None, create_time=None)