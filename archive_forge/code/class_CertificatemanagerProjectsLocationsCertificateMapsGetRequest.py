from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificateMapsGetRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsCertificateMapsGetRequest object.

  Fields:
    name: Required. A name of the certificate map to describe. Must be in the
      format `projects/*/locations/*/certificateMaps/*`.
  """
    name = _messages.StringField(1, required=True)