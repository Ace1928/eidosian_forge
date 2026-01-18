from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperationMetadata(_messages.Message):
    """The metadata associated with a long running operation resource.

  Fields:
    progressPercentage: Percentage of completion of this operation, ranging
      from 0 to 100.
    resourceNames: The full name of the resources that this operation is
      directly associated with.
    startTime: The start time of the operation.
    steps: Detailed status information for each step. The order is
      undetermined.
  """
    progressPercentage = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    resourceNames = _messages.StringField(2, repeated=True)
    startTime = _messages.StringField(3)
    steps = _messages.MessageField('Step', 4, repeated=True)