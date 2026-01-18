from oslo_config import cfg
from oslo_config import fixture
from castellan import key_manager
from castellan.key_manager import barbican_key_manager
from castellan.tests import base
class KeyManagerTestCase(base.TestCase):

    def _create_key_manager(self):
        raise NotImplementedError()

    def setUp(self):
        super(KeyManagerTestCase, self).setUp()
        self.conf = self.useFixture(fixture.Config()).conf
        self.key_mgr = self._create_key_manager()