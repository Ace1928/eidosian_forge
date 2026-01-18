from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingGetCmekSettingsRequest(_messages.Message):
    """A LoggingGetCmekSettingsRequest object.

  Fields:
    name: Required. The resource for which to retrieve CMEK settings.
      "projects/[PROJECT_ID]/cmekSettings"
      "organizations/[ORGANIZATION_ID]/cmekSettings"
      "billingAccounts/[BILLING_ACCOUNT_ID]/cmekSettings"
      "folders/[FOLDER_ID]/cmekSettings" For
      example:"organizations/12345/cmekSettings"Note: CMEK for the Log Router
      can be configured for Google Cloud projects, folders, organizations, and
      billing accounts. Once configured for an organization, it applies to all
      projects and folders in the Google Cloud organization.
  """
    name = _messages.StringField(1, required=True)