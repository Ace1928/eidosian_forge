from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateAuthorityServiceConfig(_messages.Message):
    """Contains information required to contact CA service.

  Fields:
    caPool: Required. A CA pool resource used to issue a certificate. The CA
      pool string has a relative resource path following the form
      "projects/{project}/locations/{location}/caPools/{ca_pool}".
  """
    caPool = _messages.StringField(1)