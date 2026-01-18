from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataTaxonomiesDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsDataTaxonomiesDeleteRequest object.

  Fields:
    etag: Optional. If the client provided etag value does not match the
      current etag value,the DeleteDataTaxonomy method returns an ABORTED
      error.
    name: Required. The resource name of the DataTaxonomy: projects/{project_n
      umber}/locations/{location_id}/dataTaxonomies/{data_taxonomy_id}
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)