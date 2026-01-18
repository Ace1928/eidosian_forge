from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsCertificate(_messages.Message):
    """TlsCertificate Resource

  Fields:
    cert: PEM representation.
    createTime: Output only. The time when the certificate was created in [RFC
      3339](https://tools.ietf.org/html/rfc3339) format, for example
      `2020-05-18T00:00:00.094Z`.
    expireTime: Output only. The time when the certificate expires in [RFC
      3339](https://tools.ietf.org/html/rfc3339) format, for example
      `2020-05-18T00:00:00.094Z`.
    serialNumber: Serial number, as extracted from the certificate.
    sha1Fingerprint: Sha1 Fingerprint of the certificate.
  """
    cert = _messages.StringField(1)
    createTime = _messages.StringField(2)
    expireTime = _messages.StringField(3)
    serialNumber = _messages.StringField(4)
    sha1Fingerprint = _messages.StringField(5)