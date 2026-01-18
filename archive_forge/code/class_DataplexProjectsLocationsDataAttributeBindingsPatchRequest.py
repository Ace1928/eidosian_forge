from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataAttributeBindingsPatchRequest(_messages.Message):
    """A DataplexProjectsLocationsDataAttributeBindingsPatchRequest object.

  Fields:
    googleCloudDataplexV1DataAttributeBinding: A
      GoogleCloudDataplexV1DataAttributeBinding resource to be passed as the
      request body.
    name: Output only. The relative resource name of the Data Attribute
      Binding, of the form: projects/{project_number}/locations/{location}/dat
      aAttributeBindings/{data_attribute_binding_id}
    updateMask: Required. Mask of fields to update.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1DataAttributeBinding = _messages.MessageField('GoogleCloudDataplexV1DataAttributeBinding', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)