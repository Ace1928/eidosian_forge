from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsDnsAuthorizationsCreateRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsDnsAuthorizationsCreateRequest
  object.

  Fields:
    dnsAuthorization: A DnsAuthorization resource to be passed as the request
      body.
    dnsAuthorizationId: Required. A user-provided name of the dns
      authorization.
    parent: Required. The parent resource of the dns authorization. Must be in
      the format `projects/*/locations/*`.
  """
    dnsAuthorization = _messages.MessageField('DnsAuthorization', 1)
    dnsAuthorizationId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)