from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def TargetHttpProxyArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='target HTTP proxy', completer=TargetHttpProxiesCompleter, plural=plural, custom_plural='target HTTP proxies', required=required, global_collection='compute.targetHttpProxies', regional_collection='compute.regionTargetHttpProxies', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)