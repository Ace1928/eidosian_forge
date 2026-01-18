from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferManifest(_messages.Message):
    """Specifies where the manifest is located.

  Fields:
    location: Specifies the path to the manifest in Cloud Storage. The Google-
      managed service account for the transfer must have `storage.objects.get`
      permission for this object. An example path is
      `gs://bucket_name/path/manifest.csv`.
  """
    location = _messages.StringField(1)