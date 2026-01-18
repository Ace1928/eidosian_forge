from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import attachment
from openstack import resource
from openstack.tests.unit import base
class TestAttachment(base.TestCase):

    def setUp(self):
        super(TestAttachment, self).setUp()
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.resp.headers = {}
        self.resp.status_code = 202
        self.sess = mock.Mock(spec=adapter.Adapter)
        self.sess.get = mock.Mock()
        self.sess.post = mock.Mock(return_value=self.resp)
        self.sess.put = mock.Mock(return_value=self.resp)
        self.sess.default_microversion = '3.54'

    def test_basic(self):
        sot = attachment.Attachment(ATTACHMENT)
        self.assertEqual('attachment', sot.resource_key)
        self.assertEqual('attachments', sot.resources_key)
        self.assertEqual('/attachments', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_get)
        self.assertTrue(sot.allow_commit)
        self.assertIsNotNone(sot._max_microversion)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_create_resource(self):
        sot = attachment.Attachment(**ATTACHMENT)
        self.assertEqual(ATTACHMENT['id'], sot.id)
        self.assertEqual(ATTACHMENT['status'], sot.status)
        self.assertEqual(ATTACHMENT['instance'], sot.instance)
        self.assertEqual(ATTACHMENT['volume_id'], sot.volume_id)
        self.assertEqual(ATTACHMENT['attached_at'], sot.attached_at)
        self.assertEqual(ATTACHMENT['detached_at'], sot.detached_at)
        self.assertEqual(ATTACHMENT['attach_mode'], sot.attach_mode)
        self.assertEqual(ATTACHMENT['connection_info'], sot.connection_info)

    @mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=True)
    @mock.patch.object(resource.Resource, '_translate_response')
    def test_create_no_mode_no_instance_id(self, mock_translate, mock_mv):
        self.sess.default_microversion = '3.27'
        mock_mv.return_value = False
        sot = attachment.Attachment()
        FAKE_MODE = 'rw'
        sot.create(self.sess, volume_id=FAKE_VOL_ID, connector=CONNECTOR, instance=None, mode=FAKE_MODE)
        self.sess.post.assert_called_with('/attachments', json={'attachment': {}}, headers={}, microversion='3.27', params={'volume_id': FAKE_VOL_ID, 'connector': CONNECTOR, 'instance': None, 'mode': 'rw'})
        self.sess.default_microversion = '3.54'

    @mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=True)
    @mock.patch.object(resource.Resource, '_translate_response')
    def test_create_with_mode_with_instance_id(self, mock_translate, mock_mv):
        sot = attachment.Attachment()
        FAKE_MODE = 'rw'
        sot.create(self.sess, volume_id=FAKE_VOL_ID, connector=CONNECTOR, instance=FAKE_INSTANCE_UUID, mode=FAKE_MODE)
        self.sess.post.assert_called_with('/attachments', json={'attachment': {}}, headers={}, microversion='3.54', params={'volume_id': FAKE_VOL_ID, 'connector': CONNECTOR, 'instance': FAKE_INSTANCE_UUID, 'mode': FAKE_MODE})

    @mock.patch.object(resource.Resource, '_translate_response')
    def test_complete(self, mock_translate):
        sot = attachment.Attachment()
        sot.id = FAKE_ID
        sot.complete(self.sess)
        self.sess.post.assert_called_with('/attachments/%s/action' % FAKE_ID, json={'os-complete': '92dc3671-d0ab-4370-8058-c88a71661ec5'}, microversion='3.54')