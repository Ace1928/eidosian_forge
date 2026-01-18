from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1beta2File(_messages.Message):
    """Files store content that is potentially associated with Packages or
  Versions.

  Fields:
    createTime: Output only. The time when the File was created.
    hashes: The hashes of the file content.
    name: The name of the file, for example: "projects/p1/locations/us-
      central1/repositories/repo1/files/a%2Fb%2Fc.txt". If the file ID part
      contains slashes, they are escaped.
    owner: The name of the Package or Version that owns this file, if any.
    sizeBytes: The size of the File in bytes.
    updateTime: Output only. The time when the File was last updated.
  """
    createTime = _messages.StringField(1)
    hashes = _messages.MessageField('Hash', 2, repeated=True)
    name = _messages.StringField(3)
    owner = _messages.StringField(4)
    sizeBytes = _messages.IntegerField(5)
    updateTime = _messages.StringField(6)