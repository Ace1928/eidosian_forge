from openstack.tests.unit import test_proxy_base
from openstack.workflow.v2 import _proxy
from openstack.workflow.v2 import cron_trigger
from openstack.workflow.v2 import execution
from openstack.workflow.v2 import workflow
class TestCronTriggerProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super().setUp()
        self.proxy = _proxy.Proxy(self.session)

    def test_cron_triggers(self):
        self.verify_list(self.proxy.cron_triggers, cron_trigger.CronTrigger)

    def test_cron_trigger_get(self):
        self.verify_get(self.proxy.get_cron_trigger, cron_trigger.CronTrigger)

    def test_cron_trigger_create(self):
        self.verify_create(self.proxy.create_cron_trigger, cron_trigger.CronTrigger)

    def test_cron_trigger_delete(self):
        self.verify_delete(self.proxy.delete_cron_trigger, cron_trigger.CronTrigger, True)

    def test_cron_trigger_find(self):
        self.verify_find(self.proxy.find_cron_trigger, cron_trigger.CronTrigger, expected_kwargs={'all_projects': False})