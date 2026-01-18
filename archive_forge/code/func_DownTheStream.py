from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import times
import six
def DownTheStream(cluster_upgrade):
    """Recursively gets information for the downstream Scopes."""
    downstream_state = cluster_upgrade.get('state', None)
    downstream_scopes = downstream_state.downstreamScopes if downstream_state else None
    if not downstream_scopes:
        return [cluster_upgrade]
    downstream_scope_name = downstream_scopes[0]
    if downstream_scope_name in visited:
        return [cluster_upgrade]
    visited.add(downstream_scope_name)
    downstream_scope_project = DescribeCommand.GetProjectFromScopeName(downstream_scope_name)
    downstream_feature = feature if downstream_scope_project == current_project else self.GetFeature(project=downstream_scope_project)
    downstream_cluster_upgrade = self.GetClusterUpgradeInfoForScope(downstream_scope_name, downstream_feature)
    return [cluster_upgrade] + DownTheStream(downstream_cluster_upgrade)