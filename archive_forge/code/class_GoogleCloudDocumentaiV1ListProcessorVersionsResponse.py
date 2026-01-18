from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ListProcessorVersionsResponse(_messages.Message):
    """Response message for the ListProcessorVersions method.

  Fields:
    nextPageToken: Points to the next processor, otherwise empty.
    processorVersions: The list of processors.
  """
    nextPageToken = _messages.StringField(1)
    processorVersions = _messages.MessageField('GoogleCloudDocumentaiV1ProcessorVersion', 2, repeated=True)