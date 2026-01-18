from unittest import mock
from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import policy
from heat.common import wsgi
from heat.tests import common
class DummyController(object):
    REQUEST_SCOPE = 'test'

    @util.registered_policy_enforce
    def an_action(self, req):
        return 'woot'