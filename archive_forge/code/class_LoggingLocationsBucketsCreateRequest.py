from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingLocationsBucketsCreateRequest(_messages.Message):
    """A LoggingLocationsBucketsCreateRequest object.

  Fields:
    bucketId: Required. A client-assigned identifier such as "my-bucket".
      Identifiers are limited to 100 characters and can include only letters,
      digits, underscores, hyphens, and periods. Bucket identifiers must start
      with an alphanumeric character.
    logBucket: A LogBucket resource to be passed as the request body.
    parent: Required. The resource in which to create the log bucket:
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]" For
      example:"projects/my-project/locations/global"
  """
    bucketId = _messages.StringField(1)
    logBucket = _messages.MessageField('LogBucket', 2)
    parent = _messages.StringField(3, required=True)