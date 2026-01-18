from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
def ValidateCertificateArgs(certificate_id, certificate_management):
    if certificate_management and certificate_management.upper() == 'AUTOMATIC' and certificate_id:
        raise exceptions.InvalidArgumentException('certificate-id', NO_CERTIFICATE_ID_MESSAGE)