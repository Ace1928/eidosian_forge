from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsSavedQueriesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsSavedQueriesListRequest object.

  Fields:
    filter: The standard list filter.
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order. Use "desc" after a field name for descending.
    pageSize: The standard list page size.
    pageToken: The standard list page token.
    parent: Required. The resource name of the Dataset to list SavedQueries
      from. Format:
      `projects/{project}/locations/{location}/datasets/{dataset}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)