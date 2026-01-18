from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesGetRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntrie
  sGetRequest object.

  Fields:
    name: Required. A name of the certificate map entry to describe. Must be
      in the format
      `projects/*/locations/*/certificateMaps/*/certificateMapEntries/*`.
  """
    name = _messages.StringField(1, required=True)