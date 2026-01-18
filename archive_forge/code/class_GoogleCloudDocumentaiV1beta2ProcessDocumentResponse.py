from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2ProcessDocumentResponse(_messages.Message):
    """Response to a single document processing request.

  Fields:
    inputConfig: Information about the input file. This is the same as the
      corresponding input config in the request.
    outputConfig: The output location of the parsed responses. The responses
      are written to this location as JSON-serialized `Document` objects.
  """
    inputConfig = _messages.MessageField('GoogleCloudDocumentaiV1beta2InputConfig', 1)
    outputConfig = _messages.MessageField('GoogleCloudDocumentaiV1beta2OutputConfig', 2)