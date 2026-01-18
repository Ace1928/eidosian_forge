from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def DeleteCluster(self, cluster_ref):
    """Delete a running cluster.

    Args:
      cluster_ref: cluster Resource to describe

    Returns:
      Cluster message.
    Raises:
      Error: if cluster cannot be found or caller is missing permissions. Will
        attempt to find similar clusters in other zones for a more useful error
        if the user has list permissions.
    """
    try:
        operation = self.client.projects_locations_clusters.Delete(self.messages.ContainerProjectsLocationsClustersDeleteRequest(name=ProjectLocationCluster(cluster_ref.projectId, cluster_ref.zone, cluster_ref.clusterId)))
        return self.ParseOperation(operation.name, cluster_ref.zone)
    except apitools_exceptions.HttpNotFoundError as error:
        api_error = exceptions.HttpException(error, util.HTTP_ERROR_FORMAT)
        self.CheckClusterOtherZones(cluster_ref, api_error)