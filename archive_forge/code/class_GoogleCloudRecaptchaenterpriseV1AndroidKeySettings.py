from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1AndroidKeySettings(_messages.Message):
    """Settings specific to keys that can be used by Android apps.

  Fields:
    allowAllPackageNames: Optional. If set to true, allowed_package_names are
      not enforced.
    allowedPackageNames: Optional. Android package names of apps allowed to
      use the key. Example: 'com.companyname.appname'
    supportNonGoogleAppStoreDistribution: Optional. Set to true for keys that
      are used in an Android application that is available for download in app
      stores in addition to the Google Play Store.
  """
    allowAllPackageNames = _messages.BooleanField(1)
    allowedPackageNames = _messages.StringField(2, repeated=True)
    supportNonGoogleAppStoreDistribution = _messages.BooleanField(3)