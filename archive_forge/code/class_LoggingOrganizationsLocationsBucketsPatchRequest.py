from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingOrganizationsLocationsBucketsPatchRequest(_messages.Message):
    """A LoggingOrganizationsLocationsBucketsPatchRequest object.

  Fields:
    logBucket: A LogBucket resource to be passed as the request body.
    name: Required. The full resource name of the bucket to update.
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]" "org
      anizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]
      " "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/buckets/
      [BUCKET_ID]"
      "folders/[FOLDER_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]" For
      example:"projects/my-project/locations/global/buckets/my-bucket"
    updateMask: Required. Field mask that specifies the fields in bucket that
      need an update. A bucket field will be overwritten if, and only if, it
      is in the update mask. name and output only fields cannot be updated.For
      a detailed FieldMask definition, see:
      https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#google.protobuf.FieldMaskFor
      example: updateMask=retention_days
  """
    logBucket = _messages.MessageField('LogBucket', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)