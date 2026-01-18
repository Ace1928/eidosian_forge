from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.exceptions import Error
import six
def GetInstantSnapshotRequestMessage(self):
    return self._messages.ComputeRegionInstantSnapshotsGetRequest(**self._ips_ref.AsDict())