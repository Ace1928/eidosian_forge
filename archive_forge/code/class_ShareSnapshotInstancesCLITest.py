from manilaclient.tests.functional.osc import base
class ShareSnapshotInstancesCLITest(base.OSCClientTestBase):

    def test_openstack_share_snapshot_instance_list(self):
        share = self.create_share()
        snapshot = self.create_snapshot(share['id'])
        share_snapshot_instances_list = self.listing_result('share snapshot instance', 'list --detailed')
        self.assertTableStruct(share_snapshot_instances_list, ['ID', 'Snapshot ID', 'Status', 'Created At', 'Updated At', 'Share ID', 'Share Instance ID', 'Progress', 'Provider Location'])
        self.assertIn(snapshot['id'], [item['Snapshot ID'] for item in share_snapshot_instances_list])

    def test_openstack_share_snapshot_instance_show(self):
        share = self.create_share()
        snapshot = self.create_snapshot(share['id'], wait=True)
        share_snapshot_instance = self.listing_result('share snapshot instance', f'list --snapshot {snapshot['id']}')
        result = self.dict_result('share snapshot instance', f'show {share_snapshot_instance[0]['ID']}')
        self.assertEqual(share_snapshot_instance[0]['ID'], result['id'])
        self.assertEqual(share_snapshot_instance[0]['Snapshot ID'], result['snapshot_id'])
        listing_result = self.listing_result('share snapshot instance', f'show {share_snapshot_instance[0]['ID']}')
        self.assertTableStruct(listing_result, ['Field', 'Value'])

    def test_openstack_share_snapshot_instance_set(self):
        share = self.create_share()
        snapshot = self.create_snapshot(share['id'], wait=True)
        share_snapshot_instance = self.listing_result('share snapshot instance', f'list --snapshot {snapshot['id']}')
        result1 = self.dict_result('share snapshot instance', f'show {share_snapshot_instance[0]['ID']}')
        self.assertEqual(share_snapshot_instance[0]['ID'], result1['id'])
        self.assertEqual(snapshot['id'], result1['snapshot_id'])
        self.assertEqual('available', result1['status'])
        self.openstack(f'share snapshot instance set {share_snapshot_instance[0]['ID']} --status error')
        result2 = self.dict_result('share snapshot instance', f'show {share_snapshot_instance[0]['ID']}')
        self.assertEqual(share_snapshot_instance[0]['ID'], result2['id'])
        self.assertEqual('error', result2['status'])