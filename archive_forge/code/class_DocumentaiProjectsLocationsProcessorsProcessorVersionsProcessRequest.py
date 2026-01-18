from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsProcessorVersionsProcessRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsProcessorVersionsProcessRequest
  object.

  Fields:
    googleCloudDocumentaiV1ProcessRequest: A
      GoogleCloudDocumentaiV1ProcessRequest resource to be passed as the
      request body.
    name: Required. The resource name of the Processor or ProcessorVersion to
      use for processing. If a Processor is specified, the server will use its
      default version. Format:
      `projects/{project}/locations/{location}/processors/{processor}`, or `pr
      ojects/{project}/locations/{location}/processors/{processor}/processorVe
      rsions/{processorVersion}`
  """
    googleCloudDocumentaiV1ProcessRequest = _messages.MessageField('GoogleCloudDocumentaiV1ProcessRequest', 1)
    name = _messages.StringField(2, required=True)