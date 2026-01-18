from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportInstanceRequest(_messages.Message):
    """Requestion options for importing looker data to an Instance

  Fields:
    gcsUri: Path to the import folder in Google Cloud Storage, in the form
      `gs://bucketName/folderName`.
  """
    gcsUri = _messages.StringField(1)