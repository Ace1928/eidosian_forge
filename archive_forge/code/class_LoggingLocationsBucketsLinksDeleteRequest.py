from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingLocationsBucketsLinksDeleteRequest(_messages.Message):
    """A LoggingLocationsBucketsLinksDeleteRequest object.

  Fields:
    name: Required. The full resource name of the link to delete. "projects/[P
      ROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/links/[LINK_ID]"
      "organizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKET
      _ID]/links/[LINK_ID]" "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[L
      OCATION_ID]/buckets/[BUCKET_ID]/links/[LINK_ID]" "folders/[FOLDER_ID]/lo
      cations/[LOCATION_ID]/buckets/[BUCKET_ID]/links/[LINK_ID]"
  """
    name = _messages.StringField(1, required=True)