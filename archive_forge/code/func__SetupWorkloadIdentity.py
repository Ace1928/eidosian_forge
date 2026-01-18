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
def _SetupWorkloadIdentity(args, cluster_ref, gke_cluster_ref):
    default_gsa_sentinel = None
    gsa_to_ksas = collections.OrderedDict()
    agent_gsa = args.properties.get('dataproc:dataproc.gke.agent.google-service-account', default_gsa_sentinel)
    gsa_to_ksas.setdefault(agent_gsa, []).append('agent')
    spark_driver_gsa = args.properties.get('dataproc:dataproc.gke.spark.driver.google-service-account', default_gsa_sentinel)
    gsa_to_ksas.setdefault(spark_driver_gsa, []).append('spark-driver')
    spark_executor_gsa = args.properties.get('dataproc:dataproc.gke.spark.executor.google-service-account', default_gsa_sentinel)
    gsa_to_ksas.setdefault(spark_executor_gsa, []).append('spark-executor')
    if default_gsa_sentinel in gsa_to_ksas:
        ksas = gsa_to_ksas.pop(default_gsa_sentinel)
        default_gsa = gke_workload_identity.DefaultDataprocDataPlaneServiceAccount.Get(gke_cluster_ref.projectsId)
        if default_gsa in gsa_to_ksas:
            gsa_to_ksas[default_gsa].extend(ksas)
        else:
            gsa_to_ksas[default_gsa] = ksas
    k8s_namespace = args.namespace or cluster_ref.clusterName
    log.debug('Setting up Workload Identity for the following GSA to KSAs %s in the "%s" namespace.', gsa_to_ksas, k8s_namespace)
    for gsa, ksas in gsa_to_ksas.items():
        gke_workload_identity.GkeWorkloadIdentity.UpdateGsaIamPolicy(project_id=gke_cluster_ref.projectsId, gsa_email=gsa, k8s_namespace=k8s_namespace, k8s_service_accounts=ksas)