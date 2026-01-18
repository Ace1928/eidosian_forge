from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet.clusterupgrade import flags as clusterupgrade_flags
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def GetWithForceEnable(self, project):
    """Gets the project's Cluster Upgrade Feature, enabling if necessary."""
    try:
        return self.hubclient.GetFeature(self.FeatureResourceName(project=project))
    except apitools_exceptions.HttpNotFoundError:
        self.Enable(self.messages.Feature())
        return self.GetFeature()