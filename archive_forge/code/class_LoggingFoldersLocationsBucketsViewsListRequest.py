from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingFoldersLocationsBucketsViewsListRequest(_messages.Message):
    """A LoggingFoldersLocationsBucketsViewsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of results to return from this
      request.Non-positive values are ignored. The presence of nextPageToken
      in the response indicates that more results might be available.
    pageToken: Optional. If present, then retrieve the next batch of results
      from the preceding call to this method. pageToken must be the value of
      nextPageToken from the previous response. The values of other method
      parameters should be identical to those in the previous call.
    parent: Required. The bucket whose views are to be listed:
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)