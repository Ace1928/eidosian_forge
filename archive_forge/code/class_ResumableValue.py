from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResumableValue(_messages.Message):
    """Supports the Resumable Media Upload protocol.

        Fields:
          multipart: True if this endpoint supports uploading multipart media.
          path: The URI path to be used for upload. Should be used in
            conjunction with the basePath property at the api-level.
        """
    multipart = _messages.BooleanField(1, default=True)
    path = _messages.StringField(2)