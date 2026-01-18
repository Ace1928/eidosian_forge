from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAuthorizedCertificatesResponse(_messages.Message):
    """Response message for AuthorizedCertificates.ListAuthorizedCertificates.

  Fields:
    certificates: The SSL certificates the user is authorized to administer.
    nextPageToken: Continuation token for fetching the next page of results.
  """
    certificates = _messages.MessageField('AuthorizedCertificate', 1, repeated=True)
    nextPageToken = _messages.StringField(2)