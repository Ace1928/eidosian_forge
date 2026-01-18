import json
import os
import testtools
import yaml
from keystoneauth1.fixture import serializer
class TestBetamaxSerializer(testtools.TestCase):
    TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ksa_betamax_test_cassette.yaml')
    TEST_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ksa_serializer_data.json')

    def setUp(self):
        super(TestBetamaxSerializer, self).setUp()
        self.serializer = serializer.YamlJsonSerializer()

    def test_deserialize(self):
        data = self.serializer.deserialize(open(self.TEST_FILE, 'r').read())
        request = data['http_interactions'][0]['request']
        self.assertEqual('http://keystoneauth-betamax.test/v2.0/tokens', request['uri'])
        payload = json.loads(request['body']['string'])
        self.assertEqual('test_tenant_name', payload['auth']['tenantName'])

    def test_serialize(self):
        data = json.loads(open(self.TEST_JSON, 'r').read())
        serialized = self.serializer.serialize(data)
        data = yaml.safe_load(serialized)
        request = data['http_interactions'][0]['request']
        self.assertEqual('http://keystoneauth-betamax.test/v2.0/tokens', request['uri'])
        payload = json.loads(request['body']['string'])
        self.assertEqual('test_tenant_name', payload['auth']['tenantName'])