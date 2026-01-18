from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingOrganizationsLocationsBucketsLinksListRequest(_messages.Message):
    """A LoggingOrganizationsLocationsBucketsLinksListRequest object.

  Fields:
    pageSize: Optional. The maximum number of results to return from this
      request.
    pageToken: Optional. If present, then retrieve the next batch of results
      from the preceding call to this method. pageToken must be the value of
      nextPageToken from the previous response.
    parent: Required. The parent resource whose links are to be listed:
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]" "org
      anizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]
      " "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/buckets/
      [BUCKET_ID]"
      "folders/[FOLDER_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)