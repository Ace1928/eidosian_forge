from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def NetworkEdgeSecurityServiceArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='network edge security service', completer=NetworkEdgeSecurityServicesCompleter, plural=plural, custom_plural='network edge security services', required=required, regional_collection='compute.networkEdgeSecurityServices')