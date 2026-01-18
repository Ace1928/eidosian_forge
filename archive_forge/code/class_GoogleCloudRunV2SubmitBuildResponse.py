from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2SubmitBuildResponse(_messages.Message):
    """Response message for submitting a Build.

  Fields:
    baseImageUri: URI of the base builder image in Artifact Registry being
      used in the build. Used to opt into automatic base image updates.
    buildOperation: Cloud Build operation to be polled via CloudBuild API.
  """
    baseImageUri = _messages.StringField(1)
    buildOperation = _messages.MessageField('GoogleLongrunningOperation', 2)