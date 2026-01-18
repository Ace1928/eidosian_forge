from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataAttributeBindingsCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsDataAttributeBindingsCreateRequest object.

  Fields:
    dataAttributeBindingId: Required. DataAttributeBinding identifier. * Must
      contain only lowercase letters, numbers and hyphens. * Must start with a
      letter. * Must be between 1-63 characters. * Must end with a number or a
      letter. * Must be unique within the Location.
    googleCloudDataplexV1DataAttributeBinding: A
      GoogleCloudDataplexV1DataAttributeBinding resource to be passed as the
      request body.
    parent: Required. The resource name of the parent data taxonomy
      projects/{project_number}/locations/{location_id}
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    dataAttributeBindingId = _messages.StringField(1)
    googleCloudDataplexV1DataAttributeBinding = _messages.MessageField('GoogleCloudDataplexV1DataAttributeBinding', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)