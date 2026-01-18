from novaclient.tests.unit.fixture_data import agents as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import agents
def _build_example_update_body(self):
    return {'para': {'url': '/yyy/yyyy/yyyy', 'version': '8.0', 'md5hash': 'add6bb58e139be103324d04d82d8f546'}}