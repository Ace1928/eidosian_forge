from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def ParseAutoscalingSettingsFromApiFormat(cluster_message) -> AutoscalingSettings:
    """Parses the autoscaling settings from the format returned by the describe command.

  The resulting object can later be passed to
  googlecloudsdk.api_lib.vmware.util.ConstructAutoscalingSettingsMessage.

  Args:
    cluster_message: cluster object with the autoscaling settings.

  Returns:
    Equivalent AutoscalingSettings istance.
  """
    if cluster_message.autoscalingSettings is None:
        return None
    autoscaling_settings = cluster_message.autoscalingSettings
    parsed_settings = AutoscalingSettings(min_cluster_node_count=autoscaling_settings.minClusterNodeCount, max_cluster_node_count=autoscaling_settings.maxClusterNodeCount, cool_down_period=autoscaling_settings.coolDownPeriod, autoscaling_policies={})
    for item in autoscaling_settings.autoscalingPolicies.additionalProperties:
        policy_name, policy_settings = (item.key, item.value)

        def _ParseThresholds(thresholds):
            if thresholds is None:
                return None
            return ScalingThresholds(scale_in=thresholds.scaleIn, scale_out=thresholds.scaleOut)
        parsed_policy = AutoscalingPolicy(node_type_id=policy_settings.nodeTypeId, scale_out_size=policy_settings.scaleOutSize, min_node_count=policy_settings.minNodeCount, max_node_count=policy_settings.maxNodeCount, cpu_thresholds=_ParseThresholds(policy_settings.cpuThresholds), granted_memory_thresholds=_ParseThresholds(policy_settings.grantedMemoryThresholds), consumed_memory_thresholds=_ParseThresholds(policy_settings.consumedMemoryThresholds), storage_thresholds=_ParseThresholds(policy_settings.storageThresholds))
        parsed_settings.autoscaling_policies[policy_name] = parsed_policy
    return parsed_settings