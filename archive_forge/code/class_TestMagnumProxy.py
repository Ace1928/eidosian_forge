from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import _proxy
from openstack.container_infrastructure_management.v1 import cluster
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack.container_infrastructure_management.v1 import service
from openstack.tests.unit import test_proxy_base
class TestMagnumProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super().setUp()
        self.proxy = _proxy.Proxy(self.session)