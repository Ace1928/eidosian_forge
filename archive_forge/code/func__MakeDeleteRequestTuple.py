from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
def _MakeDeleteRequestTuple(self):
    return (self._client.interconnectAttachments, 'Delete', self._messages.ComputeInterconnectAttachmentsDeleteRequest(project=self.ref.project, region=self.ref.region, interconnectAttachment=self.ref.Name()))