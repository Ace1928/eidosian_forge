from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def _MakeCreateRequestTuple(self, packet_mirroring):
    return (self._client.packetMirrorings, 'Insert', self._messages.ComputePacketMirroringsInsertRequest(project=self.ref.project, region=self.ref.region, packetMirroring=packet_mirroring))