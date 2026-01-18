from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import gke_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import clusters
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import gke_clusters
from googlecloudsdk.command_lib.dataproc import gke_workload_identity
from googlecloudsdk.command_lib.dataproc.gke_clusters import GkeNodePoolTargetsParser
from googlecloudsdk.core import log
@staticmethod
def _GetVirtualClusterConfig(dataproc, gke_cluster_ref, args, metastore_service_ref, history_server_cluster_ref):
    """Get dataproc virtual cluster configuration for GKE based clusters.

    Args:
      dataproc: Dataproc object that contains client, messages, and resources
      gke_cluster_ref: GKE cluster reference.
      args: Arguments parsed from argparse.ArgParser.
      metastore_service_ref: Reference to a Dataproc Metastore Service.
      history_server_cluster_ref: Reference to a Dataproc history cluster.

    Returns:
      virtual_cluster_config: Dataproc virtual cluster configuration
    """
    kubernetes_software_config = dataproc.messages.KubernetesSoftwareConfig(componentVersion=encoding.DictToAdditionalPropertyMessage({'SPARK': args.spark_engine_version}, dataproc.messages.KubernetesSoftwareConfig.ComponentVersionValue, sort_items=True))
    if args.properties:
        kubernetes_software_config.properties = encoding.DictToAdditionalPropertyMessage(args.properties, dataproc.messages.KubernetesSoftwareConfig.PropertiesValue, sort_items=True)
    pools = GkeNodePoolTargetsParser.Parse(dataproc, gke_cluster_ref.RelativeName(), args.pools)
    gke_cluster_config = dataproc.messages.GkeClusterConfig(gkeClusterTarget=gke_cluster_ref.RelativeName(), nodePoolTarget=pools)
    kubernetes_cluster_config = dataproc.messages.KubernetesClusterConfig(kubernetesNamespace=args.namespace, gkeClusterConfig=gke_cluster_config, kubernetesSoftwareConfig=kubernetes_software_config)
    metastore_config = None
    if metastore_service_ref:
        metastore_config = dataproc.messages.MetastoreConfig(dataprocMetastoreService=metastore_service_ref.RelativeName())
    spark_history_server_config = None
    if history_server_cluster_ref:
        spark_history_server_config = dataproc.messages.SparkHistoryServerConfig(dataprocCluster=history_server_cluster_ref.RelativeName())
    auxiliary_services_config = None
    if metastore_config or spark_history_server_config:
        auxiliary_services_config = dataproc.messages.AuxiliaryServicesConfig(metastoreConfig=metastore_config, sparkHistoryServerConfig=spark_history_server_config)
    virtual_cluster_config = dataproc.messages.VirtualClusterConfig(stagingBucket=args.staging_bucket, kubernetesClusterConfig=kubernetes_cluster_config, auxiliaryServicesConfig=auxiliary_services_config)
    return virtual_cluster_config