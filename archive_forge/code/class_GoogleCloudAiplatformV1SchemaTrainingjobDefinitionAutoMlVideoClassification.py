from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlVideoClassification(_messages.Message):
    """A TrainingJob that trains and uploads an AutoML Video Classification
  Model.

  Fields:
    inputs: The input parameters of this TrainingJob.
  """
    inputs = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlVideoClassificationInputs', 1)