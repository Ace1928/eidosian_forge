from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.exceptions import Error
import six
def GetSetInstantSnapshotLabelsRequestMessage(self, ips, labels):
    req = self._messages.ComputeRegionInstantSnapshotsSetLabelsRequest
    return req(project=self._ips_ref.project, resource=self._ips_ref.instantSnapshot, region=self._ips_ref.region, regionSetLabelsRequest=self._messages.RegionSetLabelsRequest(labelFingerprint=ips.labelFingerprint, labels=labels))