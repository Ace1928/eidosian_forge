from manilaclient.tests.functional.osc import base
class TransfersCLITest(base.OSCClientTestBase):

    def setUp(self):
        super(TransfersCLITest, self).setUp()
        self.share_type = self.create_share_type()

    def test_transfer_create_list_show_delete(self):
        share = self.create_share(share_type=self.share_type['name'], wait_for_status='available', client=self.user_client)
        self.create_share_transfer(share['id'], name='transfer_test')
        self._wait_for_object_status('share', share['id'], 'awaiting_transfer')
        transfers = self.listing_result('share', 'transfer list', client=self.user_client)
        self.assertTrue(len(transfers) > 0)
        self.assertTableStruct(transfers, ['ID', 'Name', 'Resource Type', 'Resource Id'])
        transfer = [transfer for transfer in transfers if transfer['Resource Id'] == share['id']]
        self.assertEqual(1, len(transfer))
        show_transfer = self.dict_result('share', f'transfer show {transfer[0]['ID']}')
        self.assertEqual(transfer[0]['ID'], show_transfer['id'])
        expected_keys = ('id', 'created_at', 'name', 'resource_type', 'resource_id', 'source_project_id', 'destination_project_id', 'accepted', 'expires_at')
        for key in expected_keys:
            self.assertIn(key, show_transfer)
        filtered_transfers = self.listing_result('share', f'transfer list --resource-id {share['id']}', client=self.user_client)
        self.assertEqual(1, len(filtered_transfers))
        self.assertEqual(show_transfer['resource_id'], share['id'])
        self.openstack(f'share transfer delete {show_transfer['id']}')
        self._wait_for_object_status('share', share['id'], 'available')

    def test_transfer_accept(self):
        share = self.create_share(share_type=self.share_type['name'], wait_for_status='available', client=self.user_client)
        transfer = self.create_share_transfer(share['id'], name='transfer_test')
        self._wait_for_object_status('share', share['id'], 'awaiting_transfer')
        self.openstack(f'share transfer accept {transfer['id']} {transfer['auth_key']}')
        self._wait_for_object_status('share', share['id'], 'available')