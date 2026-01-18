from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import frozendict
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet.clusterupgrade import flags as clusterupgrade_flags
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
def _UpTheStream(cluster_upgrade):
    """Recursively gets information for the upstream Fleets."""
    upstream_spec = cluster_upgrade.get('spec', None)
    upstream_fleets = upstream_spec.upstreamFleets if upstream_spec else None
    if not upstream_fleets:
        return [cluster_upgrade]
    upstream_fleet = upstream_fleets[0]
    if upstream_fleet in visited:
        return [cluster_upgrade]
    visited.add(upstream_fleet)
    upstream_fleet_project = Describe.GetProjectIDFromFleet(upstream_fleet)
    upstream_feature = feature if upstream_fleet_project == current_project else self.GetFeature(project=upstream_fleet_project)
    try:
        upstream_cluster_upgrade = Describe.GetClusterUpgradeInfo(upstream_fleet, upstream_feature)
    except exceptions.Error as e:
        log.warning(e)
        return [cluster_upgrade]
    return _UpTheStream(upstream_cluster_upgrade) + [cluster_upgrade]