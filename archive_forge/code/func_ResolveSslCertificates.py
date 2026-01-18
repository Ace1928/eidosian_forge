from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import scope as compute_scope
def ResolveSslCertificates(args, ssl_certificate_arg, target_https_proxy_ref, resources):
    """Parses the ssl certs that are pointed to by a Target HTTPS Proxy from args.

  This function handles parsing regional/global ssl certificates that are
  pointed to by a regional/global Target HTTPS Proxy.

  Args:
    args: The arguments provided to the target_https_proxies command.
    ssl_certificate_arg: The ResourceArgument specification for the
                         ssl_certificates argument.
    target_https_proxy_ref: The resource reference to the Target HTTPS Proxy.
                            This is obtained by parsing the Target HTTPS Proxy
                            arguments provided.
    resources: ComputeApiHolder resources.

  Returns:
    Returns the SSL Certificates resource
  """
    if not args.ssl_certificates:
        return []
    if not compute_scope.IsSpecifiedForFlag(args, 'ssl_certificates'):
        if IsRegionalTargetHttpsProxiesRef(target_https_proxy_ref):
            args.ssl_certificates_region = target_https_proxy_ref.region
        else:
            args.global_ssl_certificates = bool(args.ssl_certificates)
    return ssl_certificate_arg.ResolveAsResource(args, resources)