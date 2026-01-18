from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def _MakeDescribeRequestTuple(self):
    return (self._client.packetMirrorings, 'Get', self._messages.ComputePacketMirroringsGetRequest(region=self.ref.region, project=self.ref.project, packetMirroring=self.ref.Name()))