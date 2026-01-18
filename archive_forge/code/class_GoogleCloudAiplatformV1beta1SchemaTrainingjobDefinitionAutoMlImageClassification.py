from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlImageClassification(_messages.Message):
    """A TrainingJob that trains and uploads an AutoML Image Classification
  Model.

  Fields:
    inputs: The input parameters of this TrainingJob.
    metadata: The metadata information.
  """
    inputs = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlImageClassificationInputs', 1)
    metadata = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlImageClassificationMetadata', 2)