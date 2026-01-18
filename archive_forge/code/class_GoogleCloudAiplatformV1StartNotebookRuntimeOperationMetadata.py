from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1StartNotebookRuntimeOperationMetadata(_messages.Message):
    """Metadata information for NotebookService.StartNotebookRuntime.

  Fields:
    genericMetadata: The operation generic information.
    progressMessage: A human-readable message that shows the intermediate
      progress details of NotebookRuntime.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 1)
    progressMessage = _messages.StringField(2)