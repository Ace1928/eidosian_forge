from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificateIssuanceConfigsGetRequest(_messages.Message):
    """A
  CertificatemanagerProjectsLocationsCertificateIssuanceConfigsGetRequest
  object.

  Fields:
    name: Required. A name of the certificate issuance config to describe.
      Must be in the format
      `projects/*/locations/*/certificateIssuanceConfigs/*`.
  """
    name = _messages.StringField(1, required=True)