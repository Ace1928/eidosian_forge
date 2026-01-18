import uuid
from openstack.load_balancer.v2 import amphora
from openstack.tests.unit import base
class TestAmphoraConfig(base.TestCase):

    def test_basic(self):
        test_amp_config = amphora.AmphoraConfig()
        self.assertEqual('/octavia/amphorae/%(amphora_id)s/config', test_amp_config.base_path)
        self.assertFalse(test_amp_config.allow_create)
        self.assertFalse(test_amp_config.allow_fetch)
        self.assertTrue(test_amp_config.allow_commit)
        self.assertFalse(test_amp_config.allow_delete)
        self.assertFalse(test_amp_config.allow_list)