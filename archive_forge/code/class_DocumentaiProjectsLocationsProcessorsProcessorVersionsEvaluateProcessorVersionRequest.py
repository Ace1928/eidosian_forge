from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluateProcessorVersionRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluateProcesso
  rVersionRequest object.

  Fields:
    googleCloudDocumentaiV1EvaluateProcessorVersionRequest: A
      GoogleCloudDocumentaiV1EvaluateProcessorVersionRequest resource to be
      passed as the request body.
    processorVersion: Required. The resource name of the ProcessorVersion to
      evaluate. `projects/{project}/locations/{location}/processors/{processor
      }/processorVersions/{processorVersion}`
  """
    googleCloudDocumentaiV1EvaluateProcessorVersionRequest = _messages.MessageField('GoogleCloudDocumentaiV1EvaluateProcessorVersionRequest', 1)
    processorVersion = _messages.StringField(2, required=True)