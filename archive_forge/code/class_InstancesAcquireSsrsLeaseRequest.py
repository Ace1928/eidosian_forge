from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesAcquireSsrsLeaseRequest(_messages.Message):
    """Request to acquire an SSRS lease for an instance.

  Fields:
    acquireSsrsLeaseContext: Contains details about the acquire SSRS lease
      operation.
  """
    acquireSsrsLeaseContext = _messages.MessageField('AcquireSsrsLeaseContext', 1)