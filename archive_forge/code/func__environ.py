from unittest import mock
from oslo_config import cfg
from oslo_log import log
from oslo_messaging._drivers import common as rpc_common
import webob.exc
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.tests import utils
def _environ(self, path):
    return {'SERVER_NAME': 'server.test', 'SERVER_PORT': 8004, 'SCRIPT_NAME': '/v1', 'PATH_INFO': '/%s' % self.tenant + path, 'wsgi.url_scheme': 'http'}