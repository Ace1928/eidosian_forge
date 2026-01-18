from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryRowAccessPoliciesListRequest(_messages.Message):
    """A BigqueryRowAccessPoliciesListRequest object.

  Fields:
    datasetId: Required. Dataset ID of row access policies to list.
    pageSize: The maximum number of results to return in a single response
      page. Leverage the page tokens to iterate through the entire collection.
    pageToken: Page token, returned by a previous call, to request the next
      page of results.
    projectId: Required. Project ID of the row access policies to list.
    tableId: Required. Table ID of the table to list row access policies.
  """
    datasetId = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    tableId = _messages.StringField(5, required=True)