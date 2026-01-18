from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ApkManifest(_messages.Message):
    """An Android app manifest. See
  http://developer.android.com/guide/topics/manifest/manifest-intro.html

  Fields:
    applicationLabel: User-readable name for the application.
    intentFilters: A IntentFilter attribute.
    maxSdkVersion: Maximum API level on which the application is designed to
      run.
    metadata: Meta-data tags defined in the manifest.
    minSdkVersion: Minimum API level required for the application to run.
    packageName: Full Java-style package name for this application, e.g.
      "com.example.foo".
    services: Services contained in the tag.
    targetSdkVersion: Specifies the API Level on which the application is
      designed to run.
    usesFeature: Feature usage tags defined in the manifest.
    usesPermission: Permissions declared to be used by the application
    versionCode: Version number used internally by the app.
    versionName: Version number shown to users.
  """
    applicationLabel = _messages.StringField(1)
    intentFilters = _messages.MessageField('IntentFilter', 2, repeated=True)
    maxSdkVersion = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    metadata = _messages.MessageField('Metadata', 4, repeated=True)
    minSdkVersion = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    packageName = _messages.StringField(6)
    services = _messages.MessageField('Service', 7, repeated=True)
    targetSdkVersion = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    usesFeature = _messages.MessageField('UsesFeature', 9, repeated=True)
    usesPermission = _messages.StringField(10, repeated=True)
    versionCode = _messages.IntegerField(11)
    versionName = _messages.StringField(12)