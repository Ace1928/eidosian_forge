from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataTaxonomiesAttributesCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsDataTaxonomiesAttributesCreateRequest object.

  Fields:
    dataAttributeId: Required. DataAttribute identifier. * Must contain only
      lowercase letters, numbers and hyphens. * Must start with a letter. *
      Must be between 1-63 characters. * Must end with a number or a letter. *
      Must be unique within the DataTaxonomy.
    googleCloudDataplexV1DataAttribute: A GoogleCloudDataplexV1DataAttribute
      resource to be passed as the request body.
    parent: Required. The resource name of the parent data taxonomy projects/{
      project_number}/locations/{location_id}/dataTaxonomies/{data_taxonomy_id
      }
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    dataAttributeId = _messages.StringField(1)
    googleCloudDataplexV1DataAttribute = _messages.MessageField('GoogleCloudDataplexV1DataAttribute', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)