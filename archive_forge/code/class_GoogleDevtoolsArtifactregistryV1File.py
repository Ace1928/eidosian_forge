from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1File(_messages.Message):
    """Files store content that is potentially associated with Packages or
  Versions.

  Fields:
    createTime: Output only. The time when the File was created.
    fetchTime: Output only. The time when the last attempt to refresh the
      file's data was made. Only set when the repository is remote.
    hashes: The hashes of the file content.
    name: The name of the file, for example: "projects/p1/locations/us-
      central1/repositories/repo1/files/a%2Fb%2Fc.txt". If the file ID part
      contains slashes, they are escaped.
    owner: The name of the Package or Version that owns this file, if any.
    sizeBytes: The size of the File in bytes.
    updateTime: Output only. The time when the File was last updated.
  """
    createTime = _messages.StringField(1)
    fetchTime = _messages.StringField(2)
    hashes = _messages.MessageField('Hash', 3, repeated=True)
    name = _messages.StringField(4)
    owner = _messages.StringField(5)
    sizeBytes = _messages.IntegerField(6)
    updateTime = _messages.StringField(7)