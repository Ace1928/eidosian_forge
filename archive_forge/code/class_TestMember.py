from openstack.image.v2 import member
from openstack.tests.unit import base
class TestMember(base.TestCase):

    def test_basic(self):
        sot = member.Member()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('members', sot.resources_key)
        self.assertEqual('/images/%(image_id)s/members', sot.base_path)
        self.assertEqual('member', sot._alternate_id())
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = member.Member(**EXAMPLE)
        self.assertEqual(IDENTIFIER, sot.id)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['image_id'], sot.image_id)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)