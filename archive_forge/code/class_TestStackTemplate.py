from openstack.orchestration.v1 import stack_environment as se
from openstack.tests.unit import base
class TestStackTemplate(base.TestCase):

    def test_basic(self):
        sot = se.StackEnvironment()
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)

    def test_make_it(self):
        sot = se.StackEnvironment(**FAKE)
        self.assertEqual(FAKE['encrypted_param_names'], sot.encrypted_param_names)
        self.assertEqual(FAKE['event_sinks'], sot.event_sinks)
        self.assertEqual(FAKE['parameters'], sot.parameters)
        self.assertEqual(FAKE['parameter_defaults'], sot.parameter_defaults)
        self.assertEqual(FAKE['resource_registry'], sot.resource_registry)