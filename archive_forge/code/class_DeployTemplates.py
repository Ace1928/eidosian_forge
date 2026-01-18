from openstack.baremetal.v1 import deploy_templates
from openstack.tests.unit import base
class DeployTemplates(base.TestCase):

    def test_basic(self):
        sot = deploy_templates.DeployTemplate()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('deploy_templates', sot.resources_key)
        self.assertEqual('/deploy_templates', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)

    def test_instantiate(self):
        sot = deploy_templates.DeployTemplate(**FAKE)
        self.assertEqual(FAKE['steps'], sot.steps)
        self.assertEqual(FAKE['created_at'], sot.created_at)
        self.assertEqual(FAKE['extra'], sot.extra)
        self.assertEqual(FAKE['links'], sot.links)
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['updated_at'], sot.updated_at)
        self.assertEqual(FAKE['uuid'], sot.id)