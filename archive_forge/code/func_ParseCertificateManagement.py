from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
def ParseCertificateManagement(messages, certificate_management):
    if not certificate_management:
        return None
    else:
        return messages.SslSettings.SslManagementTypeValueValuesEnum(certificate_management.upper())