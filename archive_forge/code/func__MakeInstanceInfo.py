from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.packet_mirrorings import client
from googlecloudsdk.command_lib.compute.packet_mirrorings import flags
from googlecloudsdk.command_lib.compute.packet_mirrorings import utils
def _MakeInstanceInfo(instance):
    return messages.PacketMirroringMirroredResourceInfoInstanceInfo(url=utils.ResolveInstanceURI(pm_ref.project, instance, holder.resources))