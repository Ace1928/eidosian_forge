from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import scope as compute_scope
def ResolveTargetHttpsProxyUrlMap(args, url_map_arg, target_https_proxy_ref, resources):
    """Parses the URL map that is pointed to by a Target HTTPS Proxy from args.

  This function handles parsing a regional/global URL map that is
  pointed to by a regional/global Target HTTPS Proxy.

  Args:
    args: The arguments provided to the target_https_proxies command.
    url_map_arg: The ResourceArgument specification for the url map argument.
    target_https_proxy_ref: The resource reference to the Target HTTPS Proxy.
                            This is obtained by parsing the Target HTTPS Proxy
                            arguments provided.
    resources: ComputeApiHolder resources.

  Returns:
    Returns the URL map resource
  """
    if not compute_scope.IsSpecifiedForFlag(args, 'url_map'):
        if IsRegionalTargetHttpsProxiesRef(target_https_proxy_ref):
            args.url_map_region = target_https_proxy_ref.region
        else:
            args.global_url_map = bool(args.url_map)
    return url_map_arg.ResolveAsResource(args, resources)