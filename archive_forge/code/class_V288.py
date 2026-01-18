from urllib import parse
from oslo_utils import encodeutils
from novaclient import api_versions
from novaclient.tests.unit.fixture_data import base
class V288(V253):
    """Fixture data for the os-hypervisors 2.88 API."""
    api_version = '2.88'