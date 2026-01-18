import datetime
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import usage
class UsageTest(utils.TestCase):

    def setUp(self):
        super(UsageTest, self).setUp()
        self.cs = fakes.FakeClient(api_versions.APIVersion('2.0'))
        self.usage_type = self._get_usage_type()

    def _get_usage_type(self):
        return usage.Usage

    def test_usage_list(self, detailed=False):
        now = datetime.datetime.now()
        usages = self.cs.usage.list(now, now, detailed)
        self.assert_request_id(usages, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-simple-tenant-usage?' + 'start=%s&' % now.isoformat() + 'end=%s&' % now.isoformat() + 'detailed=%s' % int(bool(detailed)))
        for u in usages:
            self.assertIsInstance(u, usage.Usage)

    def test_usage_list_detailed(self):
        self.test_usage_list(True)

    def test_usage_get(self):
        now = datetime.datetime.now()
        u = self.cs.usage.get('tenantfoo', now, now)
        self.assert_request_id(u, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-simple-tenant-usage/tenantfoo?' + 'start=%s&' % now.isoformat() + 'end=%s' % now.isoformat())
        self.assertIsInstance(u, usage.Usage)

    def test_usage_class_get(self):
        start = '2012-01-22T19:48:41.750722'
        stop = '2012-01-22T19:48:41.750722'
        info = {'tenant_id': 'tenantfoo', 'start': start, 'stop': stop}
        u = usage.Usage(self.cs.usage, info)
        u.get()
        self.assert_request_id(u, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('GET', '/os-simple-tenant-usage/tenantfoo?start=%s&end=%s' % (start, stop))