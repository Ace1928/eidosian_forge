from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingFoldersLocationsBucketsViewsCreateRequest(_messages.Message):
    """A LoggingFoldersLocationsBucketsViewsCreateRequest object.

  Fields:
    logView: A LogView resource to be passed as the request body.
    parent: Required. The bucket in which to create the view
      `"projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]"`
      For example:"projects/my-project/locations/global/buckets/my-bucket"
    viewId: Required. A client-assigned identifier such as "my-view".
      Identifiers are limited to 100 characters and can include only letters,
      digits, underscores, hyphens, and periods.
  """
    logView = _messages.MessageField('LogView', 1)
    parent = _messages.StringField(2, required=True)
    viewId = _messages.StringField(3)