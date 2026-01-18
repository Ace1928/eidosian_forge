from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1MigrateKeyRequest(_messages.Message):
    """The migrate key request message.

  Fields:
    skipBillingCheck: Optional. If true, skips the billing check. A reCAPTCHA
      Enterprise key or migrated key behaves differently than a reCAPTCHA
      (non-Enterprise version) key when you reach a quota limit (see
      https://cloud.google.com/recaptcha-enterprise/quotas#quota_limit). To
      avoid any disruption of your usage, we check that a billing account is
      present. If your usage of reCAPTCHA is under the free quota, you can
      safely skip the billing check and proceed with the migration. See
      https://cloud.google.com/recaptcha-enterprise/docs/billing-information.
  """
    skipBillingCheck = _messages.BooleanField(1)