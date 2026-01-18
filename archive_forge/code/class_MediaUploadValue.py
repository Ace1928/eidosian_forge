from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MediaUploadValue(_messages.Message):
    """Media upload parameters.

    Messages:
      ProtocolsValue: Supported upload protocols.

    Fields:
      accept: MIME Media Ranges for acceptable media uploads to this method.
      maxSize: Maximum size of a media upload, such as "1MB", "2GB" or "3TB".
      protocols: Supported upload protocols.
    """

    class ProtocolsValue(_messages.Message):
        """Supported upload protocols.

      Messages:
        ResumableValue: Supports the Resumable Media Upload protocol.
        SimpleValue: Supports uploading as a single HTTP request.

      Fields:
        resumable: Supports the Resumable Media Upload protocol.
        simple: Supports uploading as a single HTTP request.
      """

        class ResumableValue(_messages.Message):
            """Supports the Resumable Media Upload protocol.

        Fields:
          multipart: True if this endpoint supports uploading multipart media.
          path: The URI path to be used for upload. Should be used in
            conjunction with the basePath property at the api-level.
        """
            multipart = _messages.BooleanField(1, default=True)
            path = _messages.StringField(2)

        class SimpleValue(_messages.Message):
            """Supports uploading as a single HTTP request.

        Fields:
          multipart: True if this endpoint supports upload multipart media.
          path: The URI path to be used for upload. Should be used in
            conjunction with the basePath property at the api-level.
        """
            multipart = _messages.BooleanField(1, default=True)
            path = _messages.StringField(2)
        resumable = _messages.MessageField('ResumableValue', 1)
        simple = _messages.MessageField('SimpleValue', 2)
    accept = _messages.StringField(1, repeated=True)
    maxSize = _messages.StringField(2)
    protocols = _messages.MessageField('ProtocolsValue', 3)