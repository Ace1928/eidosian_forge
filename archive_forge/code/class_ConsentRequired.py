from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class ConsentRequired(OAuth2Error):
    """
    The Authorization Server requires End-User consent.

    This error MAY be returned when the prompt parameter value in the
    Authentication Request is none, but the Authentication Request cannot be
    completed without displaying a user interface for End-User consent.
    """
    error = 'consent_required'