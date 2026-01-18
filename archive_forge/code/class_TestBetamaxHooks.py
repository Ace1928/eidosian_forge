import json
from unittest import mock
import betamax
from requests import models
import testtools
from keystoneauth1.fixture import hooks
class TestBetamaxHooks(testtools.TestCase):

    def test_pre_record_hook_v3(self):
        fixtures_path = 'keystoneauth1/tests/unit/data'
        with betamax.Betamax.configure() as config:
            config.before_record(callback=hooks.pre_record_hook)
        cassette = betamax.cassette.Cassette('test_pre_record_hook', 'json', record_mode=None, cassette_library_dir=fixtures_path)
        r = models.Response()
        r.status_code = 200
        r.reason = 'OK'
        r.encoding = 'utf-8'
        r.headers = {}
        r.url = 'http://localhost:35357/'
        with open('%s/keystone_v3_sample_response.json' % fixtures_path) as f:
            response_content = json.loads(f.read())
        with open('%s/keystone_v3_sample_request.json' % fixtures_path) as f:
            request_content = json.loads(f.read())
        body_content = {'body': {'string': json.dumps(response_content), 'encoding': 'utf-8'}}
        betamax.util.add_urllib3_response(body_content, r, HTTPHeaderDict({'Accept': 'application/json'}))
        response = r
        r = models.Request()
        r.method = 'GET'
        r.url = 'http://localhost:35357/'
        r.headers = {}
        r.data = {}
        response.request = r.prepare()
        response.request.headers.update({'User-Agent': 'betamax/test header'})
        response.request.body = json.dumps(request_content)
        interaction = cassette.save_interaction(response, response.request)
        response_content = json.loads(interaction.data['response']['body']['string'])
        self.assertEqual(response_content['token']['expires_at'], u'9999-12-31T23:59:59Z')
        self.assertEqual(response_content['token']['project']['domain']['id'], u'dummy')
        self.assertEqual(response_content['token']['user']['domain']['id'], u'dummy')
        self.assertEqual(response_content['token']['user']['name'], u'dummy')
        request_content = json.loads(interaction.data['request']['body']['string'])
        self.assertEqual(request_content['auth']['identity']['password']['user']['domain']['id'], u'dummy')
        self.assertEqual(request_content['auth']['identity']['password']['user']['password'], u'********')

    def test_pre_record_hook_v2(self):
        fixtures_path = 'keystoneauth1/tests/unit/data'
        with betamax.Betamax.configure() as config:
            config.before_record(callback=hooks.pre_record_hook)
        cassette = betamax.cassette.Cassette('test_pre_record_hook', 'json', record_mode=None, cassette_library_dir=fixtures_path)
        r = models.Response()
        r.status_code = 200
        r.reason = 'OK'
        r.encoding = 'utf-8'
        r.headers = {}
        r.url = 'http://localhost:35357/'
        with open('%s/keystone_v2_sample_response.json' % fixtures_path) as f:
            response_content = json.loads(f.read())
        with open('%s/keystone_v2_sample_request.json' % fixtures_path) as f:
            request_content = json.loads(f.read())
        body_content = {'body': {'string': json.dumps(response_content), 'encoding': 'utf-8'}}
        betamax.util.add_urllib3_response(body_content, r, HTTPHeaderDict({'Accept': 'application/json'}))
        response = r
        r = models.Request()
        r.method = 'GET'
        r.url = 'http://localhost:35357/'
        r.headers = {}
        r.data = {}
        response.request = r.prepare()
        response.request.headers.update({'User-Agent': 'betamax/test header'})
        response.request.body = json.dumps(request_content)
        interaction = cassette.save_interaction(response, response.request)
        response_content = json.loads(interaction.data['response']['body']['string'])
        self.assertEqual(response_content['access']['token']['expires'], u'9999-12-31T23:59:59Z')
        self.assertEqual(response_content['access']['token']['tenant']['name'], u'dummy')
        self.assertEqual(response_content['access']['user']['name'], u'dummy')
        request_content = json.loads(interaction.data['request']['body']['string'])
        self.assertEqual(request_content['auth']['passwordCredentials']['password'], u'********')
        self.assertEqual(request_content['auth']['passwordCredentials']['username'], u'dummy')
        self.assertEqual(request_content['auth']['tenantName'], u'dummy')

    @mock.patch('keystoneauth1.fixture.hooks.mask_fixture_values')
    def test_pre_record_hook_empty_body(self, mask_fixture_values):
        interaction = mock.Mock()
        interaction.data = {'request': {'body': {'encoding': 'utf-8', 'string': ''}}, 'response': {'body': {'encoding': 'utf-8', 'string': ''}}}
        hooks.pre_record_hook(interaction, mock.Mock())
        self.assertFalse(mask_fixture_values.called)