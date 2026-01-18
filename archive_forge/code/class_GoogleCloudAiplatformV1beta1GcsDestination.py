from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GcsDestination(_messages.Message):
    """The Google Cloud Storage location where the output is to be written to.

  Fields:
    outputUriPrefix: Required. Google Cloud Storage URI to output directory.
      If the uri doesn't end with '/', a '/' will be automatically appended.
      The directory is created if it doesn't exist.
  """
    outputUriPrefix = _messages.StringField(1)