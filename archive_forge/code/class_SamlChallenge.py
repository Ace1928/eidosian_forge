import abc
import base64
import sys
import pyu2f.convenience.authenticator
import pyu2f.errors
import pyu2f.model
import six
from google_reauth import _helpers, errors
class SamlChallenge(ReauthChallenge):
    """Challenge that asks the users to browse to their ID Providers."""

    @property
    def name(self):
        return 'SAML'

    @property
    def is_locally_eligible(self):
        return True

    def obtain_challenge_input(self, metadata):
        raise errors.ReauthSamlLoginRequiredError()