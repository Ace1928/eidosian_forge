from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1CheckTrialEarlyStoppingStateMetatdata(_messages.Message):
    """This message will be placed in the metadata field of a
  google.longrunning.Operation associated with a CheckTrialEarlyStoppingState
  request.

  Fields:
    genericMetadata: Operation metadata for suggesting Trials.
    study: The name of the Study that the Trial belongs to.
    trial: The Trial name.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 1)
    study = _messages.StringField(2)
    trial = _messages.StringField(3)