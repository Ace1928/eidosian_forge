from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestingApplicationDetailServiceGetApkDetailsRequest(_messages.Message):
    """A TestingApplicationDetailServiceGetApkDetailsRequest object.

  Fields:
    bundleLocation_gcsPath: A path to a file in Google Cloud Storage. Example:
      gs://build-app-1414623860166/app%40debug-unaligned.apk These paths are
      expected to be url encoded (percent encoding)
    fileReference: A FileReference resource to be passed as the request body.
  """
    bundleLocation_gcsPath = _messages.StringField(1)
    fileReference = _messages.MessageField('FileReference', 2)