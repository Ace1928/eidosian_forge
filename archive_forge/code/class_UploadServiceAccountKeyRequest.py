from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UploadServiceAccountKeyRequest(_messages.Message):
    """The service account key upload request.

  Fields:
    publicKeyData: The public key to associate with the service account. Must
      be an RSA public key that is wrapped in an X.509 v3 certificate. Include
      the first line, `-----BEGIN CERTIFICATE-----`, and the last line,
      `-----END CERTIFICATE-----`.
  """
    publicKeyData = _messages.BytesField(1)