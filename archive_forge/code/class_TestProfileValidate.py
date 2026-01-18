from openstack.clustering.v1 import profile
from openstack.tests.unit import base
class TestProfileValidate(base.TestCase):

    def setUp(self):
        super(TestProfileValidate, self).setUp()

    def test_basic(self):
        sot = profile.ProfileValidate()
        self.assertEqual('profile', sot.resource_key)
        self.assertEqual('profiles', sot.resources_key)
        self.assertEqual('/profiles/validate', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertFalse(sot.allow_list)
        self.assertEqual('PUT', sot.commit_method)