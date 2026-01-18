from openstack.clustering.v1 import profile
from openstack.tests.unit import base
class TestProfile(base.TestCase):

    def setUp(self):
        super(TestProfile, self).setUp()

    def test_basic(self):
        sot = profile.Profile()
        self.assertEqual('profile', sot.resource_key)
        self.assertEqual('profiles', sot.resources_key)
        self.assertEqual('/profiles', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)

    def test_instantiate(self):
        sot = profile.Profile(**FAKE)
        self.assertEqual(FAKE['id'], sot.id)
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['metadata'], sot.metadata)
        self.assertEqual(FAKE['spec'], sot.spec)
        self.assertEqual(FAKE['project'], sot.project_id)
        self.assertEqual(FAKE['domain'], sot.domain_id)
        self.assertEqual(FAKE['user'], sot.user_id)
        self.assertEqual(FAKE['type'], sot.type)
        self.assertEqual(FAKE['created_at'], sot.created_at)
        self.assertEqual(FAKE['updated_at'], sot.updated_at)