from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreviewArtifacts(_messages.Message):
    """Artifacts created by preview.

  Fields:
    artifacts: Output only. Location of artifacts in Google Cloud Storage.
      Format: `gs://{bucket}/{object}`
    content: Output only. Location of a blueprint copy and other content in
      Google Cloud Storage. Format: `gs://{bucket}/{object}`
  """
    artifacts = _messages.StringField(1)
    content = _messages.StringField(2)