from __future__ import unicode_literals
from oauthlib.oauth2.rfc6749.errors import FatalClientError, OAuth2Error
class AccountSelectionRequired(OpenIDClientError):
    """
    The End-User is REQUIRED to select a session at the Authorization Server.

    The End-User MAY be authenticated at the Authorization Server with
    different associated accounts, but the End-User did not select a session.
    This error MAY be returned when the prompt parameter value in the
    Authentication Request is none, but the Authentication Request cannot be
    completed without displaying a user interface to prompt for a session to
    use.
    """
    error = 'account_selection_required'