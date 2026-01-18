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
@staticmethod
def GetClusterUpgradeInfo(fleet, feature):
    """Gets Cluster Upgrade Feature information for the provided Fleet."""
    fleet_spec = feature.spec.clusterupgrade
    if not fleet_spec:
        msg = 'Cluster Upgrade feature is not configured for Fleet: {}.'.format(fleet)
        raise exceptions.Error(msg)
    res = {'fleet': fleet, 'spec': Describe.FormatDurations(fleet_spec)}
    if feature.state is not None and feature.state.clusterupgrade is not None:
        res['state'] = feature.state.clusterupgrade
    return res