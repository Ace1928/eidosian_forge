import abc
import base64
import sys
import pyu2f.convenience.authenticator
import pyu2f.errors
import pyu2f.model
import six
from google_reauth import _helpers, errors
class PasswordChallenge(ReauthChallenge):
    """Challenge that asks for user's password."""

    @property
    def name(self):
        return 'PASSWORD'

    @property
    def is_locally_eligible(self):
        return True

    def obtain_challenge_input(self, unused_metadata):
        passwd = _helpers.get_user_password('Please enter your password:')
        if not passwd:
            passwd = ' '
        return {'credential': passwd}