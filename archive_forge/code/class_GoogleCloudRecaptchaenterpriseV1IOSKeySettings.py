from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1IOSKeySettings(_messages.Message):
    """Settings specific to keys that can be used by iOS apps.

  Fields:
    allowAllBundleIds: Optional. If set to true, allowed_bundle_ids are not
      enforced.
    allowedBundleIds: Optional. iOS bundle ids of apps allowed to use the key.
      Example: 'com.companyname.productname.appname'
    appleDeveloperId: Optional. Apple Developer account details for the app
      that is protected by the reCAPTCHA Key. reCAPTCHA Enterprise leverages
      platform-specific checks like Apple App Attest and Apple DeviceCheck to
      protect your app from abuse. Providing these fields allows reCAPTCHA
      Enterprise to get a better assessment of the integrity of your app.
  """
    allowAllBundleIds = _messages.BooleanField(1)
    allowedBundleIds = _messages.StringField(2, repeated=True)
    appleDeveloperId = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1AppleDeveloperId', 3)