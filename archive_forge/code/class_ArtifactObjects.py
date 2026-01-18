from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactObjects(_messages.Message):
    """Files in the workspace to upload to Cloud Storage upon successful
  completion of all build steps.

  Fields:
    location: Cloud Storage bucket and optional object path, in the form
      "gs://bucket/path/to/somewhere/". (see [Bucket Name
      Requirements](https://cloud.google.com/storage/docs/bucket-
      naming#requirements)). Files in the workspace matching any path pattern
      will be uploaded to Cloud Storage with this location as a prefix.
    paths: Path globs used to match files in the build's workspace.
    timing: Output only. Stores timing information for pushing all artifact
      objects.
  """
    location = _messages.StringField(1)
    paths = _messages.StringField(2, repeated=True)
    timing = _messages.MessageField('TimeSpan', 3)