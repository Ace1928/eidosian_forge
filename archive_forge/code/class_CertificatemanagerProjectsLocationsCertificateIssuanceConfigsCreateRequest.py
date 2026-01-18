from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificateIssuanceConfigsCreateRequest(_messages.Message):
    """A
  CertificatemanagerProjectsLocationsCertificateIssuanceConfigsCreateRequest
  object.

  Fields:
    certificateIssuanceConfig: A CertificateIssuanceConfig resource to be
      passed as the request body.
    certificateIssuanceConfigId: Required. A user-provided name of the
      certificate config.
    parent: Required. The parent resource of the certificate issuance config.
      Must be in the format `projects/*/locations/*`.
  """
    certificateIssuanceConfig = _messages.MessageField('CertificateIssuanceConfig', 1)
    certificateIssuanceConfigId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)