from unittest import mock
from oslo_config import cfg
from oslo_log import log
from oslo_messaging._drivers import common as rpc_common
import webob.exc
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.tests import utils
def _abandon(self, path):
    return self._simple_request(path, method='DELETE')