from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def BuildGKEDestinationMessage(self, destination_gke_cluster, destination_gke_location, destination_gke_namespace, destination_gke_service, destination_gke_path):
    """Builds a Destination message for a destination GKE service.

    Args:
      destination_gke_cluster: str or None, the destination GKE service's
        cluster.
      destination_gke_location: str or None, the location of the destination GKE
        service's cluster.
      destination_gke_namespace: str or None, the destination GKE service's
        namespace.
      destination_gke_service: str or None, the destination GKE service.
      destination_gke_path: str or None, the path on the destination GKE
        service.

    Returns:
      A Destination message for a destination GKE service.
    """
    gke_message = self._messages.GKE(cluster=destination_gke_cluster, location=destination_gke_location, namespace=destination_gke_namespace, service=destination_gke_service, path=destination_gke_path)
    return self._messages.Destination(gke=gke_message)