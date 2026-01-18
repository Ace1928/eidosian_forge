from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import clusters
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _Clusters(self, args):
    """Get the clusters configs from command arguments.

    Args:
      args: the argparse namespace from Run().

    Returns:
      A dict mapping from cluster id to msg.Cluster.
    """
    msgs = util.GetAdminMessages()
    storage_type = msgs.Cluster.DefaultStorageTypeValueValuesEnum(args.cluster_storage_type.upper())
    if args.cluster_config is not None:
        if args.cluster is not None or args.cluster_zone is not None or args.cluster_num_nodes is not None:
            raise exceptions.InvalidArgumentException('--cluster-config --cluster --cluster-zone --cluster-num-nodes', 'Use --cluster-config or the combination of --cluster, --cluster-zone and --cluster-num-nodes to specify cluster(s), not both.')
        self._ValidateClusterConfigArgs(args.cluster_config)
        new_clusters = {}
        for cluster_dict in args.cluster_config:
            nodes = cluster_dict.get('nodes', 1)
            cluster = msgs.Cluster(serveNodes=nodes, defaultStorageType=storage_type, location=util.LocationUrl(cluster_dict['zone']))
            if 'kms-key' in cluster_dict:
                cluster.encryptionConfig = msgs.EncryptionConfig(kmsKeyName=cluster_dict['kms-key'])
            if 'autoscaling-min-nodes' in cluster_dict or 'autoscaling-max-nodes' in cluster_dict or 'autoscaling-cpu-target' in cluster_dict:
                if 'autoscaling-storage-target' in cluster_dict:
                    storage_target = cluster_dict['autoscaling-storage-target']
                else:
                    storage_target = None
                cluster.clusterConfig = clusters.BuildClusterConfig(autoscaling_min=cluster_dict['autoscaling-min-nodes'], autoscaling_max=cluster_dict['autoscaling-max-nodes'], autoscaling_cpu_target=cluster_dict['autoscaling-cpu-target'], autoscaling_storage_target=storage_target)
                cluster.serveNodes = None
            new_clusters[cluster_dict['id']] = cluster
        return new_clusters
    elif args.cluster is not None:
        if args.cluster_zone is None:
            raise exceptions.InvalidArgumentException('--cluster-zone', '--cluster-zone must be specified.')
        cluster = msgs.Cluster(serveNodes=arguments.ProcessInstanceTypeAndNodes(args), defaultStorageType=storage_type, location=util.LocationUrl(args.cluster_zone))
        return {args.cluster: cluster}
    else:
        raise exceptions.InvalidArgumentException('--cluster --cluster-config', 'Use --cluster-config to specify cluster(s).')