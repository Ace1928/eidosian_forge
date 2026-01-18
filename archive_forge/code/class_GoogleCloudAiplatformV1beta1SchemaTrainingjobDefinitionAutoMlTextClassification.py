from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTextClassification(_messages.Message):
    """A TrainingJob that trains and uploads an AutoML Text Classification
  Model.

  Fields:
    inputs: The input parameters of this TrainingJob.
  """
    inputs = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTextClassificationInputs', 1)