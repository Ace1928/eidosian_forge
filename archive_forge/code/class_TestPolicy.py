from openstack.clustering.v1 import policy
from openstack.tests.unit import base
class TestPolicy(base.TestCase):

    def setUp(self):
        super(TestPolicy, self).setUp()

    def test_basic(self):
        sot = policy.Policy()
        self.assertEqual('policy', sot.resource_key)
        self.assertEqual('policies', sot.resources_key)
        self.assertEqual('/policies', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_instantiate(self):
        sot = policy.Policy(**FAKE)
        self.assertEqual(FAKE['id'], sot.id)
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['spec'], sot.spec)
        self.assertEqual(FAKE['project'], sot.project_id)
        self.assertEqual(FAKE['domain'], sot.domain_id)
        self.assertEqual(FAKE['user'], sot.user_id)
        self.assertEqual(FAKE['data'], sot.data)
        self.assertEqual(FAKE['created_at'], sot.created_at)
        self.assertEqual(FAKE['updated_at'], sot.updated_at)