from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataTaxonomiesAttributesDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsDataTaxonomiesAttributesDeleteRequest object.

  Fields:
    etag: Optional. If the client provided etag value does not match the
      current etag value, the DeleteDataAttribute method returns an ABORTED
      error response.
    name: Required. The resource name of the DataAttribute: projects/{project_
      number}/locations/{location_id}/dataTaxonomies/{dataTaxonomy}/attributes
      /{data_attribute_id}
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)