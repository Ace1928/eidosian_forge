from oslo_serialization import jsonutils
from manilaclient.tests.functional.osc import base
class SharesGroupTypeAccessCLITest(base.OSCClientTestBase):

    def test_share_group_type_access_create(self):
        share_group_type_name = self.create_share_group_type(share_types='dhss_false', public=False)['name']
        access_list = self.listing_result('share', f'group type access list {share_group_type_name}')
        self.assertEqual(0, len(access_list))
        cmd_output = jsonutils.loads(self.openstack('token issue -f json '))
        auth_project_id = cmd_output['project_id']
        self.share_group_type_access_create(share_group_type_name, auth_project_id)
        share_group_type_access_list = self.listing_result('share', f'group type access list {share_group_type_name}')
        self.assertEqual(1, len(share_group_type_access_list))

    def test_share_group_type_access_delete(self):
        share_group_type_name = self.create_share_group_type(share_types='dhss_false', public=False)['name']
        access_list = self.listing_result('share', f'group type access list {share_group_type_name}')
        self.assertEqual(0, len(access_list))
        cmd_output = jsonutils.loads(self.openstack('token issue -f json '))
        auth_project_id = cmd_output['project_id']
        self.share_group_type_access_create(share_group_type_name, auth_project_id)
        share_group_type_access_list = self.listing_result('share', f'group type access list {share_group_type_name}')
        self.assertEqual(1, len(share_group_type_access_list))
        self.assertEqual(share_group_type_access_list[0]['Project ID'], auth_project_id)
        self.share_group_type_access_delete(share_group_type_name, auth_project_id)
        access_list = self.listing_result('share', f'group type access list {share_group_type_name}')
        self.assertEqual(0, len(access_list))

    def test_share_group_type_access_list(self):
        share_group_type_name = self.create_share_group_type(share_types='dhss_false', public=False)['name']
        access_list = self.listing_result('share', f'group type access list {share_group_type_name}')
        self.assertEqual(0, len(access_list))
        cmd_output = jsonutils.loads(self.openstack('token issue -f json '))
        auth_project_id = cmd_output['project_id']
        self.share_group_type_access_create(share_group_type_name, auth_project_id)
        share_group_type_access_list = self.listing_result('share', f'group type access list {share_group_type_name}')
        self.assertEqual(1, len(share_group_type_access_list))
        self.assertTableStruct(access_list, ['Project ID'])