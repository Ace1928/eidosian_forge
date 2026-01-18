from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def ExtractGkeClusterLocationId(env_object):
    """Finds the location ID of the GKE cluster running the provided environment.

  Args:
    env_object: Environment, the environment, likely returned by an API call,
      whose cluster location to extract

  Raises:
    Error: if Kubernetes cluster is not found.

  Returns:
    str, the location ID (a short name like us-central1-b) of the GKE cluster
    running the environment
  """
    if env_object.config.nodeConfig.location:
        return env_object.config.nodeConfig.location[env_object.config.nodeConfig.location.rfind('/') + 1:]
    gke_cluster = env_object.config.gkeCluster[env_object.config.gkeCluster.rfind('/') + 1:]
    gke_api = gke_api_adapter.NewAPIAdapter(GKE_API_VERSION)
    cluster_zones = [c.location[c.location.rfind('/') + 1:] or c.zone for c in gke_api.ListClusters(parsers.GetProject()).clusters if c.name == gke_cluster]
    if not cluster_zones:
        raise Error('Kubernetes Engine cluster not found.')
    elif len(cluster_zones) == 1:
        return cluster_zones[0]
    return cluster_zones[console_io.PromptChoice(['[{}]'.format(z) for z in cluster_zones], default=0, message='Cluster found in more than one location. Please select the desired location:')]