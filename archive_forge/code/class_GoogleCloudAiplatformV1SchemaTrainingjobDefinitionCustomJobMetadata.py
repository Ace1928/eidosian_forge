from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionCustomJobMetadata(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionCustomJobMetadata
  object.

  Fields:
    backingCustomJob: The resource name of the CustomJob that has been created
      to carry out this custom task.
  """
    backingCustomJob = _messages.StringField(1)