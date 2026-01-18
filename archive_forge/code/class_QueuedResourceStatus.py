from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueuedResourceStatus(_messages.Message):
    """[Output only] Result of queuing and provisioning based on deferred
  capacity.

  Fields:
    failedData: Additional status detail for the FAILED state.
    provisioningOperations: [Output only] Fully qualified URL of the
      provisioning GCE operation to track the provisioning along with
      provisioning errors. The referenced operation may not exist after having
      been deleted or expired.
    queuingPolicy: Constraints for the time when the resource(s) start
      provisioning. Always exposed as absolute times.
  """
    failedData = _messages.MessageField('QueuedResourceStatusFailedData', 1)
    provisioningOperations = _messages.StringField(2, repeated=True)
    queuingPolicy = _messages.MessageField('QueuingPolicy', 3)