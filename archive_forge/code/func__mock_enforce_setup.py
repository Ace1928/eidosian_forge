from unittest import mock
from oslo_config import cfg
from oslo_log import log
from oslo_messaging._drivers import common as rpc_common
import webob.exc
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.tests import utils
def _mock_enforce_setup(self, mocker, action, allowed=True, expected_request_count=1):
    self.mock_enforce = mocker
    self.action = action
    self.mock_enforce.return_value = allowed
    self.expected_request_count = expected_request_count