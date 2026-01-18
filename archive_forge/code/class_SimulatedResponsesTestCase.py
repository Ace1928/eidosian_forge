from collections import deque
from json import dumps
import tempfile
import unittest
from launchpadlib.errors import Unauthorized
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.launchpad import (
from launchpadlib.testing.helpers import NoNetworkAuthorizationEngine
class SimulatedResponsesTestCase(unittest.TestCase):
    """Test cases that give fake responses to launchpad's HTTP requests."""

    def setUp(self):
        """Clear out the list of simulated responses."""
        SimulatedResponsesLaunchpad.responses = []
        self.engine = NoNetworkAuthorizationEngine('http://api.example.com/', 'application name')

    def launchpad_with_responses(self, *responses):
        """Use simulated HTTP responses to get a Launchpad object.

        The given Response objects will be sent, in order, in response
        to launchpadlib's requests.

        :param responses: Some number of Response objects.
        :return: The Launchpad object, assuming that errors in the
            simulated requests didn't prevent one from being created.
        """
        SimulatedResponsesLaunchpad.responses = responses
        return SimulatedResponsesLaunchpad.login_with('application name', authorization_engine=self.engine)