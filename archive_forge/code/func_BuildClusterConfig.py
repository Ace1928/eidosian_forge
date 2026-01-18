from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
def BuildClusterConfig(autoscaling_min=None, autoscaling_max=None, autoscaling_cpu_target=None, autoscaling_storage_target=None):
    """Build a ClusterConfig field."""
    msgs = util.GetAdminMessages()
    return msgs.ClusterConfig(clusterAutoscalingConfig=BuildClusterAutoscalingConfig(min_nodes=autoscaling_min, max_nodes=autoscaling_max, cpu_target=autoscaling_cpu_target, storage_target=autoscaling_storage_target))