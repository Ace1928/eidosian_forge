from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingLocationsBucketsViewsGetRequest(_messages.Message):
    """A LoggingLocationsBucketsViewsGetRequest object.

  Fields:
    name: Required. The resource name of the policy: "projects/[PROJECT_ID]/lo
      cations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID]" For
      example:"projects/my-project/locations/global/buckets/my-
      bucket/views/my-view"
  """
    name = _messages.StringField(1, required=True)