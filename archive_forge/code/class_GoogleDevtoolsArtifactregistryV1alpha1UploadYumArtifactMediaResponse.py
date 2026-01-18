from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1alpha1UploadYumArtifactMediaResponse(_messages.Message):
    """The response to upload an artifact.

  Fields:
    operation: Operation to be returned to the user.
  """
    operation = _messages.MessageField('Operation', 1)