from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UploadKfpArtifactRequest(_messages.Message):
    """The request to upload an artifact.

  Fields:
    description: Description of the package version.
    tags: Tags to be created with the version.
  """
    description = _messages.StringField(1)
    tags = _messages.StringField(2, repeated=True)