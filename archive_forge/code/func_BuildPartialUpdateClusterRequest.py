from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
def BuildPartialUpdateClusterRequest(msgs, name=None, nodes=None, autoscaling_min=None, autoscaling_max=None, autoscaling_cpu_target=None, autoscaling_storage_target=None, update_mask=None):
    """Build a PartialUpdateClusterRequest."""
    cluster = msgs.Cluster(name=name, serveNodes=nodes)
    if autoscaling_min is not None or autoscaling_max is not None or autoscaling_cpu_target is not None or (autoscaling_storage_target is not None):
        cluster.clusterConfig = BuildClusterConfig(autoscaling_min=autoscaling_min, autoscaling_max=autoscaling_max, autoscaling_cpu_target=autoscaling_cpu_target, autoscaling_storage_target=autoscaling_storage_target)
    return msgs.BigtableadminProjectsInstancesClustersPartialUpdateClusterRequest(cluster=cluster, name=name, updateMask=update_mask)