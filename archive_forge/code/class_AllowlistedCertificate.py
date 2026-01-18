from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowlistedCertificate(_messages.Message):
    """Defines an allowlisted certificate.

  Fields:
    pemCertificate: Required. PEM certificate that is allowlisted. The
      certificate can be up to 5k bytes, and must be a parseable X.509
      certificate.
  """
    pemCertificate = _messages.StringField(1)