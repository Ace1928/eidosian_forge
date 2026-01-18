from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
class ClientTestCase(utils.ClientTestCaseMixin, TestCase):
    scenarios = [('original', {'client_fixture_class': client_fixtures.OriginalV2}), ('ksc-session', {'client_fixture_class': client_fixtures.KscSessionV2}), ('ksa-session', {'client_fixture_class': client_fixtures.KsaSessionV2})]