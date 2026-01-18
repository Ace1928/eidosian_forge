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
def AddClusterUpgradeInfoToScope(self, scope, scope_name, feature):
    serialized_scope = resource_projector.MakeSerializable(scope)
    serialized_scope['clusterUpgrade'] = self.GetClusterUpgradeInfoForScope(scope_name, feature)
    return serialized_scope