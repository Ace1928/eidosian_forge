from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def MakeDeleteRequestTuple(self):
    return (self._client.packetMirrorings, 'Delete', self._messages.ComputePacketMirroringsDeleteRequest(region=self.ref.region, project=self.ref.project, packetMirroring=self.ref.Name()))