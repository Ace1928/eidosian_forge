from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.certificate_manager import api_client
from googlecloudsdk.core.util import times
def UpdateTrustConfigAllowlistedCertificates(ref, args, request):
    """Updates allowlisted certificates based on the used flag.

  Args:
    ref: reference to the membership object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    del ref
    if not args:
        return request
    if not args.IsSpecified('add_allowlisted_certificates') and (not args.IsSpecified('remove_allowlisted_certificates')) and (not args.IsSpecified('clear_allowlisted_certificates')):
        return request
    if request.updateMask.find('allowlistedCertificates') == -1:
        if request.updateMask:
            request.updateMask += ','
        request.updateMask += 'allowlistedCertificates'
    client = api_client.GetClientInstance()
    service = client.projects_locations_trustConfigs
    messages = client.MESSAGES_MODULE
    get_trust_config_request = messages.CertificatemanagerProjectsLocationsTrustConfigsGetRequest(name=request.name)
    request.trustConfig.allowlistedCertificates = service.Get(get_trust_config_request).allowlistedCertificates
    if args.IsSpecified('remove_allowlisted_certificates'):
        pem_certificates_to_be_removed = set([NormalizePemCertificate(ac['pemCertificate']) for ac in args.remove_allowlisted_certificates if 'pemCertificate' in ac])
        request.trustConfig.allowlistedCertificates = [ac for ac in request.trustConfig.allowlistedCertificates if NormalizePemCertificate(ac.pemCertificate) not in pem_certificates_to_be_removed]
    if args.IsSpecified('clear_allowlisted_certificates'):
        request.trustConfig.allowlistedCertificates = []
    if args.IsSpecified('add_allowlisted_certificates'):
        request.trustConfig.allowlistedCertificates = request.trustConfig.allowlistedCertificates + args.add_allowlisted_certificates
    return request