from openstack.instance_ha.v1 import _proxy
from openstack.instance_ha.v1 import host
from openstack.instance_ha.v1 import notification
from openstack.instance_ha.v1 import segment
from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import test_proxy_base
class TestInstanceHaHosts(TestInstanceHaProxy):

    def test_hosts(self):
        self.verify_list(self.proxy.hosts, host.Host, method_args=[SEGMENT_ID], expected_args=[], expected_kwargs={'segment_id': SEGMENT_ID})

    def test_host_get(self):
        self.verify_get(self.proxy.get_host, host.Host, method_args=[HOST_ID], method_kwargs={'segment_id': SEGMENT_ID}, expected_kwargs={'segment_id': SEGMENT_ID})

    def test_host_create(self):
        self.verify_create(self.proxy.create_host, host.Host, method_args=[SEGMENT_ID], method_kwargs={}, expected_args=[], expected_kwargs={'segment_id': SEGMENT_ID})

    def test_host_update(self):
        self.verify_update(self.proxy.update_host, host.Host, method_kwargs={'segment_id': SEGMENT_ID})

    def test_host_delete(self):
        self.verify_delete(self.proxy.delete_host, host.Host, True, method_kwargs={'segment_id': SEGMENT_ID}, expected_kwargs={'segment_id': SEGMENT_ID})