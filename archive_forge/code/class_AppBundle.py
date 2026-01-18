from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AppBundle(_messages.Message):
    """An Android App Bundle file format, containing a BundleConfig.pb file, a
  base module directory, zero or more dynamic feature module directories. See
  https://developer.android.com/guide/app-bundle/build for guidance on
  building App Bundles.

  Fields:
    bundleLocation: .aab file representing the app bundle under test.
  """
    bundleLocation = _messages.MessageField('FileReference', 1)