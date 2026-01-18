from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTables(_messages.Message):
    """A TrainingJob that trains and uploads an AutoML Tables Model.

  Fields:
    inputs: The input parameters of this TrainingJob.
    metadata: The metadata information.
  """
    inputs = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputs', 1)
    metadata = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesMetadata', 2)