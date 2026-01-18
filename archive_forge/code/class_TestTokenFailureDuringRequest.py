from collections import deque
from json import dumps
import tempfile
import unittest
from launchpadlib.errors import Unauthorized
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.launchpad import (
from launchpadlib.testing.helpers import NoNetworkAuthorizationEngine
class TestTokenFailureDuringRequest(SimulatedResponsesTestCase):
    """Test access token failures during a request.

    launchpadlib makes two HTTP requests on startup, to get the WADL
    and JSON representations of the service root. If Launchpad
    receives a 401 error during this process, it will acquire a fresh
    access token and try again.
    """

    def test_good_token(self):
        """If our token is good, we never get another one."""
        SimulatedResponsesLaunchpad.responses = [Response(200, SIMPLE_WADL), Response(200, SIMPLE_JSON)]
        self.assertEqual(self.engine.access_tokens_obtained, 0)
        SimulatedResponsesLaunchpad.login_with('application name', authorization_engine=self.engine)
        self.assertEqual(self.engine.access_tokens_obtained, 1)

    def test_bad_token(self):
        """If our token is bad, we get another one."""
        SimulatedResponsesLaunchpad.responses = [Response(401, b'Invalid token.'), Response(200, SIMPLE_WADL), Response(200, SIMPLE_JSON)]
        self.assertEqual(self.engine.access_tokens_obtained, 0)
        SimulatedResponsesLaunchpad.login_with('application name', authorization_engine=self.engine)
        self.assertEqual(self.engine.access_tokens_obtained, 2)

    def test_expired_token(self):
        """If our token is expired, we get another one."""
        SimulatedResponsesLaunchpad.responses = [Response(401, b'Expired token.'), Response(200, SIMPLE_WADL), Response(200, SIMPLE_JSON)]
        self.assertEqual(self.engine.access_tokens_obtained, 0)
        SimulatedResponsesLaunchpad.login_with('application name', authorization_engine=self.engine)
        self.assertEqual(self.engine.access_tokens_obtained, 2)

    def test_unknown_token(self):
        """If our token is unknown, we get another one."""
        SimulatedResponsesLaunchpad.responses = [Response(401, b'Unknown access token.'), Response(200, SIMPLE_WADL), Response(200, SIMPLE_JSON)]
        self.assertEqual(self.engine.access_tokens_obtained, 0)
        SimulatedResponsesLaunchpad.login_with('application name', authorization_engine=self.engine)
        self.assertEqual(self.engine.access_tokens_obtained, 2)

    def test_delayed_error(self):
        """We get another token no matter when the error happens."""
        SimulatedResponsesLaunchpad.responses = [Response(200, SIMPLE_WADL), Response(401, b'Expired token.'), Response(200, SIMPLE_JSON)]
        self.assertEqual(self.engine.access_tokens_obtained, 0)
        SimulatedResponsesLaunchpad.login_with('application name', authorization_engine=self.engine)
        self.assertEqual(self.engine.access_tokens_obtained, 2)

    def test_many_errors(self):
        """We'll keep getting new tokens as long as tokens are the problem."""
        SimulatedResponsesLaunchpad.responses = [Response(401, b'Invalid token.'), Response(200, SIMPLE_WADL), Response(401, b'Expired token.'), Response(401, b'Invalid token.'), Response(200, SIMPLE_JSON)]
        self.assertEqual(self.engine.access_tokens_obtained, 0)
        SimulatedResponsesLaunchpad.login_with('application name', authorization_engine=self.engine)
        self.assertEqual(self.engine.access_tokens_obtained, 4)

    def test_other_unauthorized(self):
        """If the token is not at fault, a 401 error raises an exception."""
        SimulatedResponsesLaunchpad.responses = [Response(401, b'Some other error.')]
        self.assertRaises(Unauthorized, SimulatedResponsesLaunchpad.login_with, 'application name', authorization_engine=self.engine)