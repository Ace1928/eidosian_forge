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
def UpTheStream(cluster_upgrade):
    """Recursively gets information for the upstream Scopes."""
    upstream_spec = cluster_upgrade.get('spec', None)
    upstream_scopes = upstream_spec.upstreamScopes if upstream_spec else None
    if not upstream_scopes:
        return [cluster_upgrade]
    upstream_scope_name = upstream_scopes[0]
    if upstream_scope_name in visited:
        return [cluster_upgrade]
    visited.add(upstream_scope_name)
    upstream_scope_project = DescribeCommand.GetProjectFromScopeName(upstream_scope_name)
    upstream_feature = feature if upstream_scope_project == current_project else self.GetFeature(project=upstream_scope_project)
    try:
        upstream_cluster_upgrade = self.GetClusterUpgradeInfoForScope(upstream_scope_name, upstream_feature)
    except exceptions.Error as e:
        log.warning(e)
        return [cluster_upgrade]
    return UpTheStream(upstream_cluster_upgrade) + [cluster_upgrade]