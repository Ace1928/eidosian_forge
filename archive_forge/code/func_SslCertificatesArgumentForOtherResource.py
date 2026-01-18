from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def SslCertificatesArgumentForOtherResource(resource, required=True, include_regional_ssl_certificates=True):
    return compute_flags.ResourceArgument(name='--ssl-certificates', resource_name='ssl certificate', completer=SslCertificatesCompleterBeta if include_regional_ssl_certificates else SslCertificatesCompleter, plural=True, required=required, global_collection='compute.sslCertificates', regional_collection='compute.regionSslCertificates' if include_regional_ssl_certificates else None, region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION if include_regional_ssl_certificates else None, short_help='A reference to SSL certificate resources that are used for server-side authentication.', detailed_help='        References to at most 15 SSL certificate resources that are used for\n        server-side authentication. The first SSL certificate in this list is\n        considered the primary SSL certificate associated with the load\n        balancer. The SSL certificates must exist and cannot be deleted while\n        referenced by a {0}.\n        '.format(resource))