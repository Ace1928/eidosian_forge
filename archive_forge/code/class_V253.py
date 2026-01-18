from urllib import parse
from oslo_utils import encodeutils
from novaclient import api_versions
from novaclient.tests.unit.fixture_data import base
class V253(V1):
    """Fixture data for the os-hypervisors 2.53 API."""
    api_version = '2.53'
    hyper_id_1 = 'd480b1b6-2255-43c2-b2c2-d60d42c2c074'
    hyper_id_2 = '43a8214d-f36a-4fc0-a25c-3cf35c17522d'
    service_id_1 = 'a87743ff-9c29-42ff-805d-2444659b5fc0'
    service_id_2 = '0486ab8b-1cfc-4ccb-9d94-9f22ec8bbd6b'