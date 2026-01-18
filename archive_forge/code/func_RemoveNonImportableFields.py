from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import clusters
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core.util import files
def RemoveNonImportableFields(cluster):
    """Modifies cluster to exclude OUTPUT_ONLY and resource-identifying fields."""
    cluster.projectId = None
    cluster.clusterName = None
    cluster.status = None
    cluster.statusHistory = []
    cluster.clusterUuid = None
    cluster.metrics = None
    if cluster.config is not None:
        config = cluster.config
        if config.lifecycleConfig is not None:
            config.lifecycleConfig.idleStartTime = None
            config.lifecycleConfig.autoDeleteTime = None
        instance_group_configs = [config.masterConfig, config.workerConfig, config.secondaryWorkerConfig]
        for aux_config in config.auxiliaryNodeGroups:
            instance_group_configs.append(aux_config.nodeGroup.nodeGroupConfig)
        for group in instance_group_configs:
            if group is not None:
                group.instanceNames = []
                group.isPreemptible = None
                group.managedGroupConfig = None