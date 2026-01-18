from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingGetSettingsRequest(_messages.Message):
    """A LoggingGetSettingsRequest object.

  Fields:
    name: Required. The resource for which to retrieve settings.
      "projects/[PROJECT_ID]/settings"
      "organizations/[ORGANIZATION_ID]/settings"
      "billingAccounts/[BILLING_ACCOUNT_ID]/settings"
      "folders/[FOLDER_ID]/settings" For
      example:"organizations/12345/settings"Note: Settings can be retrieved
      for Google Cloud projects, folders, organizations, and billing accounts.
  """
    name = _messages.StringField(1, required=True)