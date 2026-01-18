from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsDnsAuthorizationsGetRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsDnsAuthorizationsGetRequest object.

  Fields:
    name: Required. A name of the dns authorization to describe. Must be in
      the format `projects/*/locations/*/dnsAuthorizations/*`.
  """
    name = _messages.StringField(1, required=True)