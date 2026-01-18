from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.certificate_manager import api_client
Updates a certificate map.

    Used for updating labels and description.

    Args:
      map_ref: a Resource reference to a
        certificatemanager.projects.locations.certificateMaps resource.
      labels: unified GCP Labels for the resource.
      description: str, new description

    Returns:
      Operation: the long running operation to patch certificate map.
    