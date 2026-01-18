from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingBillingAccountsLocationsBucketsLinksCreateRequest(_messages.Message):
    """A LoggingBillingAccountsLocationsBucketsLinksCreateRequest object.

  Fields:
    link: A Link resource to be passed as the request body.
    linkId: Required. The ID to use for the link. The link_id can have up to
      100 characters. A valid link_id must only have alphanumeric characters
      and underscores within it.
    parent: Required. The full resource name of the bucket to create a link
      for. "projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]"
      "organizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKET
      _ID]" "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/buck
      ets/[BUCKET_ID]"
      "folders/[FOLDER_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]"
  """
    link = _messages.MessageField('Link', 1)
    linkId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)