from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageConfig(_messages.Message):
    """The configuration for data storage in the environment.

  Fields:
    bucket: Optional. The name of the Cloud Storage bucket used by the
      environment. No `gs://` prefix.
    filestoreDirectory: Optional. The path to the Filestore directory used by
      the environment considering the share name as a root
    filestoreInstance: Optional. The Filestore instance uri used by the
      environment.
      projects/{project}/locations/{location}/instances/{instance}
  """
    bucket = _messages.StringField(1)
    filestoreDirectory = _messages.StringField(2)
    filestoreInstance = _messages.StringField(3)