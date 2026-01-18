from openstack.placement.v1 import _proxy
from openstack.placement.v1 import resource_class
from openstack.placement.v1 import resource_provider
from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import test_proxy_base as test_proxy_base
class TestPlacementProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super().setUp()
        self.proxy = _proxy.Proxy(self.session)