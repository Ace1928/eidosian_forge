from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IssueRedirectTicketInternalRequest(_messages.Message):
    """IssueRedirectTicketInternalRequest is the request to issue a redirect
  ticket for an instance. For internal use only.

  Fields:
    redirectUri: Required. URI to be used in the redirect.
  """
    redirectUri = _messages.StringField(1)