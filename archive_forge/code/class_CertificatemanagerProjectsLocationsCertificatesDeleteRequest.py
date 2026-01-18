from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificatesDeleteRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsCertificatesDeleteRequest object.

  Fields:
    name: Required. A name of the certificate to delete. Must be in the format
      `projects/*/locations/*/certificates/*`.
  """
    name = _messages.StringField(1, required=True)