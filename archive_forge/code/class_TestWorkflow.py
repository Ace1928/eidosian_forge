from openstack.tests.unit import base
from openstack.workflow.v2 import workflow
class TestWorkflow(base.TestCase):

    def setUp(self):
        super(TestWorkflow, self).setUp()

    def test_basic(self):
        sot = workflow.Workflow()
        self.assertEqual('workflow', sot.resource_key)
        self.assertEqual('workflows', sot.resources_key)
        self.assertEqual('/workflows', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)

    def test_instantiate(self):
        sot = workflow.Workflow(**FAKE)
        self.assertEqual(FAKE['id'], sot.id)
        self.assertEqual(FAKE['scope'], sot.scope)
        self.assertEqual(FAKE['definition'], sot.definition)