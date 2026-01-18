from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def UrlMapArgumentForTargetProxy(required=True, proxy_type='HTTP'):
    return compute_flags.ResourceArgument(name='--url-map', resource_name='URL map', completer=UrlMapsCompleter, plural=False, required=required, global_collection='compute.urlMaps', regional_collection='compute.regionUrlMaps', short_help='A reference to a URL map resource that defines the mapping of URLs to backend services.', detailed_help='        A reference to a URL map resource. A URL map defines the mapping of URLs\n        to backend services. Before you can refer to a URL map, you must\n        create the URL map. To delete a URL map that a target proxy is referring\n        to, you must first delete the target {0} proxy.\n        '.format(proxy_type))