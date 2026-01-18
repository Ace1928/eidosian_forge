from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingOrganizationsLocationsBucketsLinksGetRequest(_messages.Message):
    """A LoggingOrganizationsLocationsBucketsLinksGetRequest object.

  Fields:
    name: Required. The resource name of the link: "projects/[PROJECT_ID]/loca
      tions/[LOCATION_ID]/buckets/[BUCKET_ID]/links/[LINK_ID]" "organizations/
      [ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/links/[LIN
      K_ID]" "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/buc
      kets/[BUCKET_ID]/links/[LINK_ID]" "folders/[FOLDER_ID]/locations/[LOCATI
      ON_ID]/buckets/[BUCKET_ID]/links/[LINK_ID]"
  """
    name = _messages.StringField(1, required=True)