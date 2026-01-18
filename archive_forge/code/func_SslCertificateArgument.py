from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def SslCertificateArgument(required=True, plural=False, include_regional_ssl_certificates=True, global_help_text=None):
    return compute_flags.ResourceArgument(resource_name='SSL certificate', completer=SslCertificatesCompleterBeta if include_regional_ssl_certificates else SslCertificatesCompleter, plural=plural, required=required, global_collection='compute.sslCertificates', global_help_text=global_help_text, regional_collection='compute.regionSslCertificates' if include_regional_ssl_certificates else None, region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION if include_regional_ssl_certificates else None)