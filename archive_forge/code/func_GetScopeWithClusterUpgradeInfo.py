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
def GetScopeWithClusterUpgradeInfo(self, scope, feature):
    """Adds Cluster Upgrade Feature information to describe Scope response."""
    scope_name = ClusterUpgradeCommand.GetScopeNameWithProjectNumber(scope.name)
    if self.args.IsKnownAndSpecified('show_cluster_upgrade') and self.args.show_cluster_upgrade:
        return self.AddClusterUpgradeInfoToScope(scope, scope_name, feature)
    elif self.args.IsKnownAndSpecified('show_linked_cluster_upgrade') and self.args.show_linked_cluster_upgrade:
        serialized_scope = resource_projector.MakeSerializable(scope)
        serialized_scope['clusterUpgrades'] = self.GetLinkedClusterUpgradeScopes(scope_name, feature)
        return serialized_scope
    return scope