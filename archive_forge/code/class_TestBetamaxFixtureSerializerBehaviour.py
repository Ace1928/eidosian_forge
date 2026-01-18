from unittest import mock
import betamax
from betamax import exceptions
import testtools
from keystoneauth1.fixture import keystoneauth_betamax
from keystoneauth1.fixture import serializer
from keystoneauth1.fixture import v2 as v2Fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
class TestBetamaxFixtureSerializerBehaviour(testtools.TestCase):
    """Test the fixture's logic, not its monkey-patching.

    The setUp method of our BetamaxFixture monkey-patches the function to
    construct a session. We don't need to test that particular bit of logic
    here so we do not need to call useFixture in our setUp method.
    """

    @mock.patch.object(betamax.Betamax, 'register_serializer')
    def test_can_pass_custom_serializer(self, register_serializer):
        serializer = mock.Mock()
        serializer.name = 'mocked-serializer'
        fixture = keystoneauth_betamax.BetamaxFixture(cassette_name='fake', cassette_library_dir='keystoneauth1/tests/unit/data', serializer=serializer)
        register_serializer.assert_called_once_with(serializer)
        self.assertIs(serializer, fixture.serializer)
        self.assertEqual('mocked-serializer', fixture.serializer_name)

    def test_can_pass_serializer_name(self):
        fixture = keystoneauth_betamax.BetamaxFixture(cassette_name='fake', cassette_library_dir='keystoneauth1/tests/unit/data', serializer_name='json')
        self.assertIsNone(fixture.serializer)
        self.assertEqual('json', fixture.serializer_name)

    def test_no_serializer_options_provided(self):
        fixture = keystoneauth_betamax.BetamaxFixture(cassette_name='fake', cassette_library_dir='keystoneauth1/tests/unit/data')
        self.assertIs(serializer.YamlJsonSerializer, fixture.serializer)
        self.assertEqual('yamljson', fixture.serializer_name)

    def test_no_request_matchers_provided(self):
        fixture = keystoneauth_betamax.BetamaxFixture(cassette_name='fake', cassette_library_dir='keystoneauth1/tests/unit/data')
        self.assertDictEqual({}, fixture.use_cassette_kwargs)

    def test_request_matchers(self):
        fixture = keystoneauth_betamax.BetamaxFixture(cassette_name='fake', cassette_library_dir='keystoneauth1/tests/unit/data', request_matchers=['method', 'uri', 'json-body'])
        self.assertDictEqual({'match_requests_on': ['method', 'uri', 'json-body']}, fixture.use_cassette_kwargs)