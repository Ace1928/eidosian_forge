from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def GetCertificateBySerialNum(ca_pool_ref, serial_num):
    """Obtains a certificate by serial num by filtering all certs in a CA pool.

  Args:
    ca_pool_ref: The resource reference to the CA pool.
    serial_num: The serial number to lookup the certificate by.

  Returns:
    The certificate message of the corresponding serial number. Ignores
    duplicate certificates.

  Raises:
    exceptions.InvalidArgumentError if there were no certificates with the
    specified CA pool and serial number.
  """
    cert_filter = 'certificate_description.subject_description.hex_serial_number:{}'.format(serial_num)
    client = base.GetClientInstance(api_version='v1')
    messages = base.GetMessagesModule(api_version='v1')
    response = client.projects_locations_caPools_certificates.List(messages.PrivatecaProjectsLocationsCaPoolsCertificatesListRequest(parent=ca_pool_ref.RelativeName(), filter=cert_filter))
    if not response.certificates:
        raise exceptions.InvalidArgumentException('--serial-number', 'The serial number specified does not exist under the CA pool [{}]]'.format(ca_pool_ref.RelativeName()))
    return response.certificates[0]