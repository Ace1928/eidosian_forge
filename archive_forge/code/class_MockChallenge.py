import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import _reauth_async
from google.oauth2 import reauth
class MockChallenge(object):

    def __init__(self, name, locally_eligible, challenge_input):
        self.name = name
        self.is_locally_eligible = locally_eligible
        self.challenge_input = challenge_input

    def obtain_challenge_input(self, metadata):
        return self.challenge_input