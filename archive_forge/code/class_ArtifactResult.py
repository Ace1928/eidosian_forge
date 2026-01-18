from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactResult(_messages.Message):
    """An artifact that was uploaded during a build. This is a single record in
  the artifact manifest JSON file.

  Fields:
    fileHash: The file hash of the artifact.
    location: The path of an artifact in a Google Cloud Storage bucket, with
      the generation number. For example,
      `gs://mybucket/path/to/output.jar#generation`.
  """
    fileHash = _messages.MessageField('FileHashes', 1, repeated=True)
    location = _messages.StringField(2)