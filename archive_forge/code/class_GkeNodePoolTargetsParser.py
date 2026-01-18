from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
class GkeNodePoolTargetsParser:
    """Parses all the --pools flags into a list of GkeNodePoolTarget messages."""

    @staticmethod
    def Parse(dataproc, gke_cluster, arg_pools, support_shuffle_service=False):
        """Parses all the --pools flags into a list of GkeNodePoolTarget messages.

    Args:
      dataproc: The Dataproc API version to use for GkeNodePoolTarget messages.
      gke_cluster: The GKE cluster's relative name, for example,
        'projects/p1/locations/l1/clusters/c1'.
      arg_pools: The list of dict[str, any] generated from all --pools flags.
      support_shuffle_service: support shuffle service.

    Returns:
      A list of GkeNodePoolTargets message, one for each entry in the arg_pools
      list.
    """
        pools = [_GkeNodePoolTargetParser.Parse(dataproc, gke_cluster, arg_pool, support_shuffle_service) for arg_pool in arg_pools]
        GkeNodePoolTargetsParser._ValidateUniqueNames(pools)
        GkeNodePoolTargetsParser._ValidateRoles(dataproc, pools)
        GkeNodePoolTargetsParser._ValidatePoolsHaveSameLocation(pools)
        GkeNodePoolTargetsParser._ValidateBootDiskKmsKeyPattern(pools)
        return pools

    @staticmethod
    def _ValidateUniqueNames(pools):
        """Validates that pools have unique names."""
        used_names = set()
        for pool in pools:
            name = pool.nodePool
            if name in used_names:
                raise exceptions.InvalidArgumentException('--pools', 'Pool name "%s" used more than once.' % name)
            used_names.add(name)

    @staticmethod
    def _ValidateRoles(dataproc, pools):
        """Validates that roles are exclusive and that one pool has DEFAULT."""
        if not pools:
            return
        seen_roles = set()
        for pool in pools:
            for role in pool.roles:
                if role in seen_roles:
                    raise exceptions.InvalidArgumentException('--pools', 'Multiple pools contained the same role "%s".' % role)
                else:
                    seen_roles.add(role)
        default = dataproc.messages.GkeNodePoolTarget.RolesValueListEntryValuesEnum('DEFAULT')
        if default not in seen_roles:
            raise exceptions.InvalidArgumentException('--pools', 'If any pools are specified, then exactly one must have the "default" role.')

    @staticmethod
    def _ValidatePoolsHaveSameLocation(pools):
        """Validates that all pools specify an identical location."""
        if not pools:
            return
        initial_locations = None
        for pool in pools:
            if pool.nodePoolConfig is not None:
                locations = pool.nodePoolConfig.locations
                if initial_locations is None:
                    initial_locations = locations
                    continue
                elif initial_locations != locations:
                    raise exceptions.InvalidArgumentException('--pools', 'All pools must have identical locations.')

    @staticmethod
    def _ValidateBootDiskKmsKeyPattern(pools):
        """Validates that the bootDiskKmsKey matches the correct pattern."""
        if not pools:
            return
        boot_disk_kms_key_pattern = re.compile('projects/[^/]+/locations/[^/]+/keyRings/[^/]+/cryptoKeys/[^/]+')
        for pool in pools:
            if pool.nodePoolConfig is None or pool.nodePoolConfig.config is None or pool.nodePoolConfig.config.bootDiskKmsKey is None:
                continue
            if not boot_disk_kms_key_pattern.match(pool.nodePoolConfig.config.bootDiskKmsKey):
                raise exceptions.InvalidArgumentException('--pools', 'bootDiskKmsKey must match pattern: projects/[KEY_PROJECT_ID]/locations/[LOCATION]/keyRings/[RING_NAME]/cryptoKeys/[KEY_NAME]')