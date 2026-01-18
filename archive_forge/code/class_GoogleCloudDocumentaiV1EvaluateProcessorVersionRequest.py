from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1EvaluateProcessorVersionRequest(_messages.Message):
    """Evaluates the given ProcessorVersion against the supplied documents.

  Fields:
    evaluationDocuments: Optional. The documents used in the evaluation. If
      unspecified, use the processor's dataset as evaluation input.
  """
    evaluationDocuments = _messages.MessageField('GoogleCloudDocumentaiV1BatchDocumentsInputConfig', 1)