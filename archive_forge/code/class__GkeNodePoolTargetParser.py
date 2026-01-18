from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
class _GkeNodePoolTargetParser:
    """Helper to parse a --pools flag into a GkeNodePoolTarget message."""
    _ARG_ROLE_TO_API_ROLE = {'default': 'DEFAULT', 'controller': 'CONTROLLER', 'spark-driver': 'SPARK_DRIVER', 'spark-executor': 'SPARK_EXECUTOR'}

    @staticmethod
    def Parse(dataproc, gke_cluster, arg_pool, support_shuffle_service=False):
        """Parses a --pools flag into a GkeNodePoolTarget message.

    Args:
      dataproc: The Dataproc API version to use for the GkeNodePoolTarget
        message.
      gke_cluster: The GKE cluster's relative name, for example,
        'projects/p1/locations/l1/clusters/c1'.
      arg_pool: The dict[str, any] generated from the --pools flag.
      support_shuffle_service: support shuffle service.

    Returns:
      A GkeNodePoolTarget message.
    """
        return _GkeNodePoolTargetParser._GkeNodePoolTargetFromArgPool(dataproc, gke_cluster, arg_pool, support_shuffle_service)

    @staticmethod
    def _GkeNodePoolTargetFromArgPool(dataproc, gke_cluster, arg_pool, support_shuffle_service=False):
        """Creates a GkeNodePoolTarget from a --pool argument."""
        return dataproc.messages.GkeNodePoolTarget(nodePool='{0}/nodePools/{1}'.format(gke_cluster, arg_pool['name']), roles=_GkeNodePoolTargetParser._SplitRoles(dataproc, arg_pool['roles'], support_shuffle_service), nodePoolConfig=_GkeNodePoolTargetParser._GkeNodePoolConfigFromArgPool(dataproc, arg_pool))

    @staticmethod
    def _SplitRoles(dataproc, arg_roles, support_shuffle_service=False):
        """Splits the role string given as an argument into a list of Role enums."""
        roles = []
        support_shuffle_service = _GkeNodePoolTargetParser._ARG_ROLE_TO_API_ROLE
        if support_shuffle_service:
            defined_roles = _GkeNodePoolTargetParser._ARG_ROLE_TO_API_ROLE.copy()
            defined_roles.update({'shuffle-service': 'SHUFFLE_SERVICE'})
        for arg_role in arg_roles.split(';'):
            if arg_role.lower() not in defined_roles:
                raise exceptions.InvalidArgumentException('--pools', 'Unrecognized role "%s".' % arg_role)
            roles.append(dataproc.messages.GkeNodePoolTarget.RolesValueListEntryValuesEnum(defined_roles[arg_role.lower()]))
        return roles

    @staticmethod
    def _GkeNodePoolConfigFromArgPool(dataproc, arg_pool):
        """Creates the GkeNodePoolConfig via the arguments specified in --pools."""
        config = dataproc.messages.GkeNodePoolConfig(config=_GkeNodePoolTargetParser._GkeNodeConfigFromArgPool(dataproc, arg_pool), autoscaling=_GkeNodePoolTargetParser._GkeNodePoolAutoscalingConfigFromArgPool(dataproc, arg_pool))
        if 'locations' in arg_pool:
            config.locations = arg_pool['locations'].split(';')
        if config != dataproc.messages.GkeNodePoolConfig():
            return config
        return None

    @staticmethod
    def _GkeNodeConfigFromArgPool(dataproc, arg_pool):
        """Creates the GkeNodeConfig via the arguments specified in --pools."""
        pool = dataproc.messages.GkeNodeConfig()
        if 'machineType' in arg_pool:
            pool.machineType = arg_pool['machineType']
        if 'preemptible' in arg_pool:
            pool.preemptible = arg_pool['preemptible']
        if 'localSsdCount' in arg_pool:
            pool.localSsdCount = arg_pool['localSsdCount']
        if 'localNvmeSsdCount' in arg_pool:
            pool.ephemeralStorageConfig = dataproc.messages.GkeEphemeralStorageConfig(localSsdCount=arg_pool['localNvmeSsdCount'])
        if 'accelerators' in arg_pool:
            pool.accelerators = _GkeNodePoolTargetParser._GkeNodePoolAcceleratorConfigFromArgPool(dataproc, arg_pool['accelerators'])
        if 'minCpuPlatform' in arg_pool:
            pool.minCpuPlatform = arg_pool['minCpuPlatform']
        if 'bootDiskKmsKey' in arg_pool:
            pool.bootDiskKmsKey = arg_pool['bootDiskKmsKey']
        if pool != dataproc.messages.GkeNodeConfig():
            return pool
        return None

    @staticmethod
    def _GkeNodePoolAcceleratorConfigFromArgPool(dataproc, arg_accelerators):
        """Creates the GkeNodePoolAcceleratorConfig via the arguments specified in --pools."""
        accelerators = []
        for arg_accelerator in arg_accelerators.split(';'):
            if '=' not in arg_accelerator:
                raise exceptions.InvalidArgumentException('--pools', 'accelerators value "%s" does not match the expected "ACCELERATOR_TYPE=ACCELERATOR_VALUE" pattern.' % arg_accelerator)
            accelerator_type, count_string = arg_accelerator.split('=', 1)
            try:
                count = int(count_string)
                accelerators.append(dataproc.messages.GkeNodePoolAcceleratorConfig(acceleratorCount=count, acceleratorType=accelerator_type))
            except ValueError:
                raise exceptions.InvalidArgumentException('--pools', 'Unable to parse accelerators count "%s" as an integer.' % count_string)
        return accelerators

    @staticmethod
    def _GkeNodePoolAutoscalingConfigFromArgPool(dataproc, arg_pool):
        """Creates the GkeNodePoolAutoscalingConfig via the arguments specified in --pools."""
        config = dataproc.messages.GkeNodePoolAutoscalingConfig()
        if 'min' in arg_pool:
            config.minNodeCount = arg_pool['min']
        if 'max' in arg_pool:
            config.maxNodeCount = arg_pool['max']
        if config != dataproc.messages.GkeNodePoolAutoscalingConfig():
            return config
        return None