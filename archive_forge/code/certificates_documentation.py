from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.certificate_manager import api_client
Updates a certificate.

    Used for updating labels, description and certificate data.

    Args:
      cert_ref: a Resource reference to a
        certificatemanager.projects.locations.certificates resource.
      self_managed_cert_data: API message for self-managed certificate data.
      labels: unified GCP Labels for the resource.
      description: str, new description

    Returns:
      Operation: the long running operation to patch certificate.
    