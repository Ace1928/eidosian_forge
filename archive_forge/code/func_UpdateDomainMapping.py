from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def UpdateDomainMapping(self, domain, certificate_id, no_certificate_id, management_type):
    """Updates a domain mapping for the given application.

    Args:
      domain: str, the custom domain string.
      certificate_id: str, a certificate id for the domain.
      no_certificate_id: bool, remove the certificate id from the domain.
      management_type: SslSettings.SslManagementTypeValueValuesEnum,
                       AUTOMATIC or MANUAL certificate provisioning.

    Returns:
      The updated DomainMapping object.
    """
    mask_fields = []
    if certificate_id or no_certificate_id:
        mask_fields.append('sslSettings.certificateId')
    if management_type:
        mask_fields.append('sslSettings.sslManagementType')
    ssl = self.messages.SslSettings(certificateId=certificate_id, sslManagementType=management_type)
    domain_mapping = self.messages.DomainMapping(id=domain, sslSettings=ssl)
    if not mask_fields:
        raise exceptions.MinimumArgumentException(['--[no-]certificate-id', '--no_managed_certificate'], 'Please specify at least one attribute to the domain-mapping update.')
    request = self.messages.AppengineAppsDomainMappingsPatchRequest(name=self._FormatDomainMapping(domain), domainMapping=domain_mapping, updateMask=','.join(mask_fields))
    operation = self.client.apps_domainMappings.Patch(request)
    return operations_util.WaitForOperation(self.client.apps_operations, operation).response