from openstack.shared_file_system.v2 import share_group as _share_group
from openstack.tests.functional.shared_file_system import base
class ShareGroupTest(base.BaseSharedFileSystemTest):

    def setUp(self):
        super(ShareGroupTest, self).setUp()
        self.SHARE_GROUP_NAME = self.getUniqueString()
        share_grp = self.user_cloud.shared_file_system.create_share_group(name=self.SHARE_GROUP_NAME)
        self.user_cloud.shared_file_system.wait_for_status(share_grp, status='available', failures=['error'], interval=5, wait=self._wait_for_timeout)
        self.assertIsNotNone(share_grp)
        self.assertIsNotNone(share_grp.id)
        self.SHARE_GROUP_ID = share_grp.id

    def test_get(self):
        sot = self.user_cloud.shared_file_system.get_share_group(self.SHARE_GROUP_ID)
        assert isinstance(sot, _share_group.ShareGroup)
        self.assertEqual(self.SHARE_GROUP_ID, sot.id)

    def test_find(self):
        sot = self.user_cloud.shared_file_system.find_share_group(self.SHARE_GROUP_NAME)
        assert isinstance(sot, _share_group.ShareGroup)
        self.assertEqual(self.SHARE_GROUP_NAME, sot.name)
        self.assertEqual(self.SHARE_GROUP_ID, sot.id)

    def test_list_delete_share_group(self):
        s_grps = self.user_cloud.shared_file_system.share_groups()
        self.assertGreater(len(list(s_grps)), 0)
        for s_grp in s_grps:
            for attribute in ('id', 'name', 'created_at'):
                self.assertTrue(hasattr(s_grp, attribute))
            sot = self.conn.shared_file_system.delete_share_group(s_grp)
            self.assertIsNone(sot)

    def test_update(self):
        u_gp = self.user_cloud.shared_file_system.update_share_group(self.SHARE_GROUP_ID, description='updated share group')
        get_u_gp = self.user_cloud.shared_file_system.get_share_group(u_gp.id)
        self.assertEqual('updated share group', get_u_gp.description)