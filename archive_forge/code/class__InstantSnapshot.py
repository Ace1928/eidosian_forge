from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.exceptions import Error
import six
class _InstantSnapshot(_CommonInstantSnapshot):
    """A wrapper for Compute Engine InstantSnapshotsService API client."""

    def __init__(self, client, ips_ref, messages):
        _CommonInstantSnapshot.__init__(self)
        self._ips_ref = ips_ref
        self._client = client
        self._service = client.instantSnapshots
        self._messages = messages

    @classmethod
    def GetOperationCollection(cls):
        return 'compute.zoneOperations'

    def GetInstantSnapshotRequestMessage(self):
        return self._messages.ComputeInstantSnapshotsGetRequest(**self._ips_ref.AsDict())

    def GetSetLabelsRequestMessage(self):
        return self._messages.ZoneSetLabelsRequest

    def GetSetInstantSnapshotLabelsRequestMessage(self, ips, labels):
        req = self._messages.ComputeInstantSnapshotsSetLabelsRequest
        return req(project=self._ips_ref.project, resource=self._ips_ref.instantSnapshot, zone=self._ips_ref.zone, zoneSetLabelsRequest=self._messages.ZoneSetLabelsRequest(labelFingerprint=ips.labelFingerprint, labels=labels))