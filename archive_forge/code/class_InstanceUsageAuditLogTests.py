from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
class InstanceUsageAuditLogTests(utils.TestCase):

    def setUp(self):
        super(InstanceUsageAuditLogTests, self).setUp()
        self.cs = fakes.FakeClient(api_versions.APIVersion('2.1'))

    def test_instance_usage_audit_log(self):
        audit_log = self.cs.instance_usage_audit_log.get()
        self.assert_request_id(audit_log, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-instance_usage_audit_log')

    def test_instance_usage_audit_log_with_before(self):
        audit_log = self.cs.instance_usage_audit_log.get(before='2016-12-10 13:59:59.999999')
        self.assert_request_id(audit_log, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-instance_usage_audit_log/2016-12-10%2013%3A59%3A59.999999')

    def test_instance_usage_audit_log_with_before_unicode(self):
        before = '\\u5de5\\u4f5c'
        self.assertRaises(exceptions.BadRequest, self.cs.instance_usage_audit_log.get, before)