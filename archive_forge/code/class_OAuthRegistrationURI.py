from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OAuthRegistrationURI(_messages.Message):
    """RPC Response object returned by GetOAuthRegistrationURL

  Fields:
    registrationUri: The URL that the user should be redirected to in order to
      start the OAuth flow. When the user is redirected to this URL, they will
      be sent to the source provider specified in the request to authorize
      CloudBuild to access their oauth credentials. After the authorization is
      completed, the user will be redirected to the Cloud Build console.
  """
    registrationUri = _messages.StringField(1)