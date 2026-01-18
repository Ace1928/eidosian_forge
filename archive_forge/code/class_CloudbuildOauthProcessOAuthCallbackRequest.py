from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildOauthProcessOAuthCallbackRequest(_messages.Message):
    """A CloudbuildOauthProcessOAuthCallbackRequest object.

  Fields:
    code: GitHub generated temproary authorization code.
    githubEnterpriseConfig: For github enterprise, the full resource name of
      the github enterprise resource.
    hostUrl: The host url of the site that the OAuth token is issued for.
    namespace: The namespace that the oauth callback credential should be
      processed for. This should map to the string name of the enum defined in
      the GetOAuthRegistrationURLRequest.
    state: The XSRF token that was sent as part of the initial request to
      start the OAuth flow.
  """
    code = _messages.StringField(1)
    githubEnterpriseConfig = _messages.StringField(2)
    hostUrl = _messages.StringField(3)
    namespace = _messages.StringField(4)
    state = _messages.StringField(5)