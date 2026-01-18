from unittest import mock
from webob import exc
from heat.api.openstack.v1 import util
from heat.common import context
from heat.common import policy
from heat.common import wsgi
from heat.tests import common
class TestPolicyEnforce(common.HeatTestCase):

    def setUp(self):
        super(TestPolicyEnforce, self).setUp()
        self.req = wsgi.Request({})
        self.req.context = context.RequestContext(project_id='foo', is_admin=False)

        class DummyController(object):
            REQUEST_SCOPE = 'test'

            @util.registered_policy_enforce
            def an_action(self, req):
                return 'woot'
        self.controller = DummyController()

    @mock.patch.object(policy.Enforcer, 'enforce')
    def test_policy_enforce_tenant_mismatch(self, mock_enforce):
        mock_enforce.return_value = True
        self.assertEqual('woot', self.controller.an_action(self.req, 'foo'))
        self.assertRaises(exc.HTTPForbidden, self.controller.an_action, self.req, tenant_id='bar')

    @mock.patch.object(policy.Enforcer, 'enforce')
    def test_policy_enforce_tenant_mismatch_is_admin(self, mock_enforce):
        self.req.context = context.RequestContext(project_id='foo', is_admin=True)
        mock_enforce.return_value = True
        self.assertEqual('woot', self.controller.an_action(self.req, 'foo'))
        self.assertEqual('woot', self.controller.an_action(self.req, 'bar'))

    @mock.patch.object(policy.Enforcer, 'enforce')
    def test_policy_enforce_policy_deny(self, mock_enforce):
        mock_enforce.return_value = False
        self.assertRaises(exc.HTTPForbidden, self.controller.an_action, self.req, tenant_id='foo')