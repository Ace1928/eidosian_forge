from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
class TestMessageProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestMessageProxy, self).setUp()
        self.proxy = _proxy.Proxy(self.session)