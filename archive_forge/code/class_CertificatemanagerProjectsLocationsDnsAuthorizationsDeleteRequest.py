from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsDnsAuthorizationsDeleteRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsDnsAuthorizationsDeleteRequest
  object.

  Fields:
    name: Required. A name of the dns authorization to delete. Must be in the
      format `projects/*/locations/*/dnsAuthorizations/*`.
  """
    name = _messages.StringField(1, required=True)