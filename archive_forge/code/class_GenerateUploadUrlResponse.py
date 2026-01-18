from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateUploadUrlResponse(_messages.Message):
    """Response of `GenerateSourceUploadUrl` method.

  Fields:
    storageSource: The location of the source code in the upload bucket. Once
      the archive is uploaded using the `upload_url` use this field to set the
      `function.build_config.source.storage_source` during CreateFunction and
      UpdateFunction. Generation defaults to 0, as Cloud Storage provides a
      new generation only upon uploading a new object or version of an object.
    uploadUrl: The generated Google Cloud Storage signed URL that should be
      used for a function source code upload. The uploaded file should be a
      zip archive which contains a function.
  """
    storageSource = _messages.MessageField('StorageSource', 1)
    uploadUrl = _messages.StringField(2)