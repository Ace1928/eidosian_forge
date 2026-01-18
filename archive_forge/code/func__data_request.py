from unittest import mock
from oslo_config import cfg
from oslo_log import log
from oslo_messaging._drivers import common as rpc_common
import webob.exc
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.tests import utils
def _data_request(self, path, data, content_type='application/json', method='POST'):
    environ = self._environ(path)
    environ['REQUEST_METHOD'] = method
    req = wsgi.Request(environ)
    req.context = utils.dummy_context('api_test_user', self.tenant)
    self.context = req.context
    req.body = data.encode('latin-1')
    return req