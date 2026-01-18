from collections import deque
from json import dumps
import tempfile
import unittest
from launchpadlib.errors import Unauthorized
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.launchpad import (
from launchpadlib.testing.helpers import NoNetworkAuthorizationEngine
class TestAbilityToParseData(SimulatedResponsesTestCase):
    """Test launchpadlib's ability to handle the sample data.

    To create a Launchpad object, two HTTP requests must succeed and
    return usable data: the requests for the WADL and JSON
    representations of the service root. This test shows that the
    minimal data in SIMPLE_WADL and SIMPLE_JSON is good enough to
    create a Launchpad object.
    """

    def test_minimal_data(self):
        """Make sure that launchpadlib can use the minimal data."""
        self.launchpad_with_responses(Response(200, SIMPLE_WADL), Response(200, SIMPLE_JSON))

    def test_bad_wadl(self):
        """Show that bad WADL causes an exception."""
        self.assertRaises(SyntaxError, self.launchpad_with_responses, Response(200, b'This is not WADL.'), Response(200, SIMPLE_JSON))

    def test_bad_json(self):
        """Show that bad JSON causes an exception."""
        self.assertRaises(JSONDecodeError, self.launchpad_with_responses, Response(200, SIMPLE_WADL), Response(200, b'This is not JSON.'))