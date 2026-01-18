from openstack.orchestration.v1 import software_deployment
from openstack.tests.unit import base
class TestSoftwareDeployment(base.TestCase):

    def test_basic(self):
        sot = software_deployment.SoftwareDeployment()
        self.assertEqual('software_deployment', sot.resource_key)
        self.assertEqual('software_deployments', sot.resources_key)
        self.assertEqual('/software_deployments', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = software_deployment.SoftwareDeployment(**FAKE)
        self.assertEqual(FAKE['id'], sot.id)
        self.assertEqual(FAKE['action'], sot.action)
        self.assertEqual(FAKE['config_id'], sot.config_id)
        self.assertEqual(FAKE['creation_time'], sot.created_at)
        self.assertEqual(FAKE['server_id'], sot.server_id)
        self.assertEqual(FAKE['stack_user_project_id'], sot.stack_user_project_id)
        self.assertEqual(FAKE['input_values'], sot.input_values)
        self.assertEqual(FAKE['output_values'], sot.output_values)
        self.assertEqual(FAKE['status'], sot.status)
        self.assertEqual(FAKE['status_reason'], sot.status_reason)
        self.assertEqual(FAKE['updated_time'], sot.updated_at)