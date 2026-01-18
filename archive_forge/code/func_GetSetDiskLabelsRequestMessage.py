from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.core.exceptions import Error
def GetSetDiskLabelsRequestMessage(self, disk, labels):
    req = self._messages.ComputeRegionDisksSetLabelsRequest
    return req(project=self._disk_ref.project, resource=self._disk_ref.disk, region=self._disk_ref.region, regionSetLabelsRequest=self._messages.RegionSetLabelsRequest(labelFingerprint=disk.labelFingerprint, labels=labels))