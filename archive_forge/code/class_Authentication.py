from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Authentication(_messages.Message):
    """`Authentication` defines the authentication configuration for an API.
  Example for an API targeted for external use:      name:
  calendar.googleapis.com     authentication:       rules:       - selector:
  "*"         oauth:           canonical_scopes:
  https://www.googleapis.com/auth/calendar        - selector:
  google.calendar.Delegate         oauth:           canonical_scopes:
  https://www.googleapis.com/auth/calendar.read

  Fields:
    providers: Defines a set of authentication providers that a service
      supports.
    rules: Individual rules for authentication.
  """
    providers = _messages.MessageField('AuthProvider', 1, repeated=True)
    rules = _messages.MessageField('AuthenticationRule', 2, repeated=True)