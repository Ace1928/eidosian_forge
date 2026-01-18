from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsDisableRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsDisableRequest object.

  Fields:
    googleCloudDocumentaiV1DisableProcessorRequest: A
      GoogleCloudDocumentaiV1DisableProcessorRequest resource to be passed as
      the request body.
    name: Required. The processor resource name to be disabled.
  """
    googleCloudDocumentaiV1DisableProcessorRequest = _messages.MessageField('GoogleCloudDocumentaiV1DisableProcessorRequest', 1)
    name = _messages.StringField(2, required=True)