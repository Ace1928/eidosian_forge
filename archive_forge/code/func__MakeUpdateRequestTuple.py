from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def _MakeUpdateRequestTuple(self, packet_mirroring):
    return (self._client.packetMirrorings, 'Patch', self._messages.ComputePacketMirroringsPatchRequest(project=self.ref.project, region=self.ref.region, packetMirroring=self.ref.Name(), packetMirroringResource=packet_mirroring))