from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ListTaxonomiesResponse(_messages.Message):
    """Response message for ListTaxonomies.

  Fields:
    nextPageToken: Pagination token of the next results page. Empty if there
      are no more results in the list.
    taxonomies: Taxonomies that the project contains.
  """
    nextPageToken = _messages.StringField(1)
    taxonomies = _messages.MessageField('GoogleCloudDatacatalogV1Taxonomy', 2, repeated=True)