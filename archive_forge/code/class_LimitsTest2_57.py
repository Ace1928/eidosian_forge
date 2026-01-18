from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import limits as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import limits
class LimitsTest2_57(LimitsTest):
    data_fixture_class = data.Fixture2_57
    supports_image_meta = False
    supports_personality = False

    def setUp(self):
        super(LimitsTest2_57, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.57')