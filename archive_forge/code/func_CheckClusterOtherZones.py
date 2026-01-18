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
def CheckClusterOtherZones(self, cluster_ref, api_error):
    """Searches for similar clusters in other locations and reports via error.

    Args:
      cluster_ref: cluster Resource to look for others with the same ID in
        different locations.
      api_error: current error from original request.

    Raises:
      Error: wrong zone error if another similar cluster found, otherwise not
      found error.
    """
    not_found_error = util.Error(NO_SUCH_CLUSTER_ERROR_MSG.format(error=api_error, name=cluster_ref.clusterId, project=cluster_ref.projectId))
    try:
        clusters = self.ListClusters(cluster_ref.projectId).clusters
    except apitools_exceptions.HttpForbiddenError as error:
        raise not_found_error
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error, util.HTTP_ERROR_FORMAT)
    for cluster in clusters:
        if cluster.name == cluster_ref.clusterId:
            if cluster.zone == cluster_ref.zone:
                raise api_error
            raise util.Error(WRONG_ZONE_ERROR_MSG.format(error=api_error, name=cluster_ref.clusterId, wrong_zone=self.Zone(cluster_ref), zone=cluster.zone))
    raise not_found_error