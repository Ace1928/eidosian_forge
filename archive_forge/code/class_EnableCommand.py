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
class EnableCommand(feature_base.EnableCommandMixin, ClusterUpgradeCommand):
    """Base class for enabling the Cluster Upgrade Feature."""

    def GetWithForceEnable(self):
        """Gets the project's Cluster Upgrade Feature, enabling if necessary."""
        try:
            return self.hubclient.GetFeature(self.FeatureResourceName())
        except apitools_exceptions.HttpNotFoundError:
            self.Enable(self.messages.Feature())
            return self.GetFeature()