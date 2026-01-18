from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def UrlMapArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(name='url_map', resource_name='URL map', completer=UrlMapsCompleter, plural=plural, required=required, global_collection='compute.urlMaps', regional_collection='compute.regionUrlMaps', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)