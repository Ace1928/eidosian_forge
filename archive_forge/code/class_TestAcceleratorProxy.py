from openstack.accelerator.v2 import _proxy
from openstack.accelerator.v2 import accelerator_request
from openstack.accelerator.v2 import deployable
from openstack.accelerator.v2 import device_profile
from openstack.tests.unit import test_proxy_base as test_proxy_base
class TestAcceleratorProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestAcceleratorProxy, self).setUp()
        self.proxy = _proxy.Proxy(self.session)