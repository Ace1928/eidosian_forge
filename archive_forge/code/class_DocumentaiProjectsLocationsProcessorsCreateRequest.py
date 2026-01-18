from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsCreateRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsCreateRequest object.

  Fields:
    googleCloudDocumentaiV1Processor: A GoogleCloudDocumentaiV1Processor
      resource to be passed as the request body.
    parent: Required. The parent (project and location) under which to create
      the processor. Format: `projects/{project}/locations/{location}`
  """
    googleCloudDocumentaiV1Processor = _messages.MessageField('GoogleCloudDocumentaiV1Processor', 1)
    parent = _messages.StringField(2, required=True)