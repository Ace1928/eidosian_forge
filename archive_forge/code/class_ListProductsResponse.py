from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListProductsResponse(_messages.Message):
    """Response message for the `ListProducts` method.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    products: List of products.
  """
    nextPageToken = _messages.StringField(1)
    products = _messages.MessageField('Product', 2, repeated=True)