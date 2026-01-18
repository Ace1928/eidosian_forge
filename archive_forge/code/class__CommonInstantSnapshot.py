from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.exceptions import Error
import six
class _CommonInstantSnapshot(six.with_metaclass(abc.ABCMeta, object)):
    """Common class for InstantSnapshot Service API client."""

    def GetService(self):
        return self._service

    def GetInstantSnapshotResource(self):
        request_msg = self.GetInstantSnapshotRequestMessage()
        return self._service.Get(request_msg)

    @abc.abstractmethod
    def GetInstantSnapshotRequestMessage(self):
        raise NotImplementedError

    @abc.abstractmethod
    def GetSetLabelsRequestMessage(self):
        raise NotImplementedError

    @abc.abstractmethod
    def GetSetInstantSnapshotLabelsRequestMessage(self):
        raise NotImplementedError