import collections
import copy
import os.path
from oslo_serialization import jsonutils
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import environments
from mistralclient.tests.unit.v2 import base
from mistralclient import utils
class TestEnvironmentsV2(base.BaseClientV2Test):

    def test_create(self):
        data = copy.deepcopy(ENVIRONMENT)
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, status_code=201, json=data)
        env = self.environments.create(**data)
        self.assertIsNotNone(env)
        expected_data = copy.deepcopy(data)
        expected_data['variables'] = jsonutils.dumps(expected_data['variables'])
        self.assertEqual(expected_data, self.requests_mock.last_request.json())

    def test_create_with_json_file_uri(self):
        path = pkg.resource_filename('mistralclient', 'tests/unit/resources/env_v2.json')
        path = os.path.abspath(path)
        uri = parse.urljoin('file:', request.pathname2url(path))
        data = collections.OrderedDict(utils.load_content(utils.get_contents_if_file(uri)))
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, status_code=201, json=data)
        file_input = {'file': uri}
        env = self.environments.create(**file_input)
        self.assertIsNotNone(env)
        expected_data = copy.deepcopy(data)
        expected_data['variables'] = jsonutils.dumps(expected_data['variables'])
        self.assertEqual(expected_data, self.requests_mock.last_request.json())

    def test_create_without_name(self):
        data = copy.deepcopy(ENVIRONMENT)
        data.pop('name')
        e = self.assertRaises(api_base.APIException, self.environments.create, **data)
        self.assertEqual(400, e.error_code)

    def test_update(self):
        data = copy.deepcopy(ENVIRONMENT)
        self.requests_mock.put(self.TEST_URL + URL_TEMPLATE, json=data)
        env = self.environments.update(**data)
        self.assertIsNotNone(env)
        expected_data = copy.deepcopy(data)
        expected_data['variables'] = jsonutils.dumps(expected_data['variables'])
        self.assertEqual(expected_data, self.requests_mock.last_request.json())

    def test_update_with_yaml_file(self):
        path = pkg.resource_filename('mistralclient', 'tests/unit/resources/env_v2.json')
        data = collections.OrderedDict(utils.load_content(utils.get_contents_if_file(path)))
        self.requests_mock.put(self.TEST_URL + URL_TEMPLATE, json=data)
        file_input = {'file': path}
        env = self.environments.update(**file_input)
        self.assertIsNotNone(env)
        expected_data = copy.deepcopy(data)
        expected_data['variables'] = jsonutils.dumps(expected_data['variables'])
        self.assertEqual(expected_data, self.requests_mock.last_request.json())

    def test_update_without_name(self):
        data = copy.deepcopy(ENVIRONMENT)
        data.pop('name')
        e = self.assertRaises(api_base.APIException, self.environments.update, **data)
        self.assertEqual(400, e.error_code)

    def test_list(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'environments': [ENVIRONMENT]})
        environment_list = self.environments.list()
        self.assertEqual(1, len(environment_list))
        env = environment_list[0]
        self.assertDictEqual(environments.Environment(self.environments, ENVIRONMENT).to_dict(), env.to_dict())

    def test_get(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE_NAME % 'env', json=ENVIRONMENT)
        env = self.environments.get('env')
        self.assertIsNotNone(env)
        self.assertDictEqual(environments.Environment(self.environments, ENVIRONMENT).to_dict(), env.to_dict())

    def test_delete(self):
        self.requests_mock.delete(self.TEST_URL + URL_TEMPLATE_NAME % 'env', status_code=204)
        self.environments.delete('env')