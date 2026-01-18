from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def PacketMirroringArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='packet mirroring', completer=PacketMirroringCompleter, plural=plural, custom_plural='packet mirrorings', required=required, regional_collection='compute.packetMirrorings')