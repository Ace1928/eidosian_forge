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
def ModifyCrossConnectSubnetworks(self, cluster_ref, existing_cross_connect_config, add_subnetworks=None, remove_subnetworks=None, clear_all_subnetworks=None):
    """Add/Remove/Clear cross connect subnetworks and schedule cluster update request.
    """
    items = existing_cross_connect_config.items
    if clear_all_subnetworks:
        items = []
    if remove_subnetworks:
        items = [x for x in items if x.subnetwork not in remove_subnetworks]
    if add_subnetworks:
        existing_subnetworks = set([x.subnetwork for x in items])
        items.extend([self.messages.CrossConnectItem(subnetwork=subnetwork) for subnetwork in add_subnetworks if subnetwork not in existing_subnetworks])
    cross_connect_config = self.messages.CrossConnectConfig(fingerprint=existing_cross_connect_config.fingerprint, items=items)
    private_cluster_config = self.messages.PrivateClusterConfig(crossConnectConfig=cross_connect_config)
    update = self.messages.ClusterUpdate(desiredPrivateClusterConfig=private_cluster_config)
    op = self.client.projects_locations_clusters.Update(self.messages.UpdateClusterRequest(name=ProjectLocationCluster(cluster_ref.projectId, cluster_ref.zone, cluster_ref.clusterId), update=update))
    return self.ParseOperation(op.name, cluster_ref.zone)