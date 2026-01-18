from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesPolicyTagsListRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesPolicyTagsListRequest object.

  Fields:
    pageSize: The maximum number of items to return. Must be a value between 1
      and 1000 inclusively. If not set, defaults to 50.
    pageToken: The pagination token of the next results page. If not set,
      returns the first page. The token is returned in the response to a
      previous list request.
    parent: Required. Resource name of the taxonomy to list the policy tags
      of.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)