from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def InterconnectRemoteLocationArgumentForOtherResource(short_help, required=False, detailed_help=None):
    return compute_flags.ResourceArgument(name='--remote-location', resource_name='interconnectRemoteLocation', completer=InterconnectRemoteLocationsCompleter, plural=False, required=required, global_collection='compute.interconnectRemoteLocations', short_help=short_help, detailed_help=detailed_help)