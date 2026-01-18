from openstack.orchestration.v1 import software_config
from openstack.tests.unit import base
class TestSoftwareConfig(base.TestCase):

    def test_basic(self):
        sot = software_config.SoftwareConfig()
        self.assertEqual('software_config', sot.resource_key)
        self.assertEqual('software_configs', sot.resources_key)
        self.assertEqual('/software_configs', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = software_config.SoftwareConfig(**FAKE)
        self.assertEqual(FAKE_ID, sot.id)
        self.assertEqual(FAKE_NAME, sot.name)
        self.assertEqual(FAKE['config'], sot.config)
        self.assertEqual(FAKE['creation_time'], sot.created_at)
        self.assertEqual(FAKE['group'], sot.group)
        self.assertEqual(FAKE['inputs'], sot.inputs)
        self.assertEqual(FAKE['outputs'], sot.outputs)
        self.assertEqual(FAKE['options'], sot.options)