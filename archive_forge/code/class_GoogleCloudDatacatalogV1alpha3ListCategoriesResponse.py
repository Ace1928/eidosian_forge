from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3ListCategoriesResponse(_messages.Message):
    """Response message for "CategoryManager.ListCategories".

  Fields:
    categories: Categories that are in this taxonomy.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    categories = _messages.MessageField('GoogleCloudDatacatalogV1alpha3Category', 1, repeated=True)
    nextPageToken = _messages.StringField(2)