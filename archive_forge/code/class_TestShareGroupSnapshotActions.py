from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share_group_snapshot
from openstack.tests.unit import base
class TestShareGroupSnapshotActions(TestShareGroupSnapshot):

    def setUp(self):
        super(TestShareGroupSnapshot, self).setUp()
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.status_code = 200
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.sess = mock.Mock(spec=adapter.Adapter)
        self.sess.default_microversion = '3.0'
        self.sess.post = mock.Mock(return_value=self.resp)
        self.sess._get_connection = mock.Mock(return_value=self.cloud)

    def test_reset_status(self):
        sot = share_group_snapshot.ShareGroupSnapshot(**EXAMPLE)
        self.assertIsNone(sot.reset_status(self.sess, 'available'))
        url = f'share-group-snapshots/{IDENTIFIER}/action'
        body = {'reset_status': {'status': 'available'}}
        headers = {'Accept': ''}
        self.sess.post.assert_called_with(url, json=body, headers=headers, microversion=self.sess.default_microversion)

    def test_get_members(self):
        sot = share_group_snapshot.ShareGroupSnapshot(**EXAMPLE)
        sot.get_members(self.sess)
        url = f'share-group-snapshots/{IDENTIFIER}/members'
        headers = {'Accept': ''}
        self.sess.get.assert_called_with(url, headers=headers, microversion=self.sess.default_microversion)