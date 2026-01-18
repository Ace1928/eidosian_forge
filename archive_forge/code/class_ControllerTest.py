from unittest import mock
from oslo_config import cfg
from oslo_log import log
from oslo_messaging._drivers import common as rpc_common
import webob.exc
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.tests import utils
class ControllerTest(object):
    """Common utilities for testing API Controllers."""

    def __init__(self, *args, **kwargs):
        super(ControllerTest, self).__init__(*args, **kwargs)
        cfg.CONF.set_default('host', 'server.test')
        self.topic = rpc_api.ENGINE_TOPIC
        self.api_version = '1.0'
        self.tenant = 't'
        self.mock_enforce = None
        log.register_options(cfg.CONF)

    def _environ(self, path):
        return {'SERVER_NAME': 'server.test', 'SERVER_PORT': 8004, 'SCRIPT_NAME': '/v1', 'PATH_INFO': '/%s' % self.tenant + path, 'wsgi.url_scheme': 'http'}

    def _simple_request(self, path, params=None, method='GET'):
        environ = self._environ(path)
        environ['REQUEST_METHOD'] = method
        if params:
            qs = '&'.join(['='.join([k, str(params[k])]) for k in params])
            environ['QUERY_STRING'] = qs
        req = wsgi.Request(environ)
        req.context = utils.dummy_context('api_test_user', self.tenant)
        self.context = req.context
        return req

    def _get(self, path, params=None):
        return self._simple_request(path, params=params)

    def _delete(self, path):
        return self._simple_request(path, method='DELETE')

    def _abandon(self, path):
        return self._simple_request(path, method='DELETE')

    def _data_request(self, path, data, content_type='application/json', method='POST'):
        environ = self._environ(path)
        environ['REQUEST_METHOD'] = method
        req = wsgi.Request(environ)
        req.context = utils.dummy_context('api_test_user', self.tenant)
        self.context = req.context
        req.body = data.encode('latin-1')
        return req

    def _post(self, path, data, content_type='application/json'):
        return self._data_request(path, data, content_type)

    def _put(self, path, data, content_type='application/json'):
        return self._data_request(path, data, content_type, method='PUT')

    def _patch(self, path, data, content_type='application/json'):
        return self._data_request(path, data, content_type, method='PATCH')

    def _url(self, id):
        host = 'server.test:8004'
        path = '/v1/%(tenant)s/stacks/%(stack_name)s/%(stack_id)s%(path)s' % id
        return 'http://%s%s' % (host, path)

    def tearDown(self):
        if self.mock_enforce:
            self.mock_enforce.assert_called_with(action=self.action, context=self.context, scope=self.controller.REQUEST_SCOPE, target={'project_id': self.tenant}, is_registered_policy=mock.ANY)
            self.assertEqual(self.expected_request_count, len(self.mock_enforce.call_args_list))
        super(ControllerTest, self).tearDown()

    def _mock_enforce_setup(self, mocker, action, allowed=True, expected_request_count=1):
        self.mock_enforce = mocker
        self.action = action
        self.mock_enforce.return_value = allowed
        self.expected_request_count = expected_request_count