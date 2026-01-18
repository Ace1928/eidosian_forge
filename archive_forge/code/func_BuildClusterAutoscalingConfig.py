from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
def BuildClusterAutoscalingConfig(min_nodes=None, max_nodes=None, cpu_target=None, storage_target=None):
    """Build a ClusterAutoscalingConfig field."""
    msgs = util.GetAdminMessages()
    limits = msgs.AutoscalingLimits(minServeNodes=min_nodes, maxServeNodes=max_nodes)
    targets = msgs.AutoscalingTargets(cpuUtilizationPercent=cpu_target, storageUtilizationGibPerNode=storage_target)
    return msgs.ClusterAutoscalingConfig(autoscalingLimits=limits, autoscalingTargets=targets)