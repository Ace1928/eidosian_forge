from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
class TestDiscoveryAPIPolicy(APIPolicyBase):

    def setUp(self):
        super(TestDiscoveryAPIPolicy, self).setUp()
        self.enforcer = mock.MagicMock()
        self.context = mock.MagicMock()
        self.policy = policy.DiscoveryAPIPolicy(self.context, enforcer=self.enforcer)

    def test_stores_info_detail(self):
        self.policy.stores_info_detail()
        self.enforcer.enforce.assert_called_once_with(self.context, 'stores_info_detail', mock.ANY)