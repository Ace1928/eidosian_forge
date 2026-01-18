from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
Updates a domain mapping for the given application.

    Args:
      domain: str, the custom domain string.
      certificate_id: str, a certificate id for the domain.
      no_certificate_id: bool, remove the certificate id from the domain.
      management_type: SslSettings.SslManagementTypeValueValuesEnum,
                       AUTOMATIC or MANUAL certificate provisioning.

    Returns:
      The updated DomainMapping object.
    