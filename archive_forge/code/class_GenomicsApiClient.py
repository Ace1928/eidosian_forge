from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.api_lib.genomics import genomics_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
import six
class GenomicsApiClient(six.with_metaclass(abc.ABCMeta, object)):
    """Base class for clients for accessing the genomics API.
  """

    def __init__(self, version):
        self._messages = genomics_util.GetGenomicsMessages(version)
        self._client = genomics_util.GetGenomicsClient(version)
        self._registry = resources.REGISTRY.Clone()
        self._registry.RegisterApiByName('genomics', version)

    @abc.abstractmethod
    def ResourceFromName(self, name):
        raise NotImplementedError()

    @abc.abstractmethod
    def Poller(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def GetOperation(self, resource):
        raise NotImplementedError()

    @abc.abstractmethod
    def CancelOperation(self, resource):
        raise NotImplementedError()