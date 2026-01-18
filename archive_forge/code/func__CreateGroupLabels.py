from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.calliope import exceptions
def _CreateGroupLabels(policy_group_labels):
    group_labels = []
    for policy_group_label in policy_group_labels or []:
        pairs = {label.key: label.value for label in policy_group_label.labels.additionalProperties}
        group_labels.append(pairs)
    return group_labels