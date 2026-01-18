from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryDatasetsListRequest(_messages.Message):
    """A BigqueryDatasetsListRequest object.

  Fields:
    all: Whether to list all datasets, including hidden ones
    filter: An expression for filtering the results of the request by label.
      The syntax is "labels.[:]". Multiple filters can be ANDed together by
      connecting with a space. Example: "labels.department:receiving
      labels.active". See https://cloud.google.com/bigquery/docs/labeling-
      datasets#filtering_datasets_using_labels for details.
    maxResults: The maximum number of results to return
    pageToken: Page token, returned by a previous call, to request the next
      page of results
    projectId: Project ID of the datasets to be listed
  """
    all = _messages.BooleanField(1)
    filter = _messages.StringField(2)
    maxResults = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(4)
    projectId = _messages.StringField(5, required=True)