from openstack.instance_ha.v1 import _proxy
from openstack.instance_ha.v1 import host
from openstack.instance_ha.v1 import notification
from openstack.instance_ha.v1 import segment
from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import test_proxy_base
class TestInstanceHaVMoves(TestInstanceHaProxy):

    def test_vmoves(self):
        self.verify_list(self.proxy.vmoves, vmove.VMove, method_args=[NOTIFICATION_ID], expected_args=[], expected_kwargs={'notification_id': NOTIFICATION_ID})

    def test_vmove_get(self):
        self.verify_get(self.proxy.get_vmove, vmove.VMove, method_args=[VMOVE_ID, NOTIFICATION_ID], expected_args=[VMOVE_ID], expected_kwargs={'notification_id': NOTIFICATION_ID})