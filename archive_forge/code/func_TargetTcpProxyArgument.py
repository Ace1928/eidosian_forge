from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def TargetTcpProxyArgument(required=True, plural=False, allow_regional=False):
    return compute_flags.ResourceArgument(resource_name='target TCP proxy', completer=TargetTcpProxiesCompleter, plural=plural, custom_plural='target TCP proxies', required=required, global_collection='compute.targetTcpProxies', regional_collection='compute.regionTargetTcpProxies' if allow_regional else None, region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)