from unittest import mock
from oslo_config import cfg
from oslotest import base
from cinderclient import exceptions as cinder_exception
from glance_store.common import attachment_state_manager as attach_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
class AttachmentStateTestCase(base.BaseTestCase):

    def setUp(self):
        super(AttachmentStateTestCase, self).setUp()
        self.attachments = set()
        self.m = attach_manager._AttachmentState()
        self.attach_call_1 = [mock.sentinel.client, mock.sentinel.volume_id]
        self.attach_call_2 = {'mode': mock.sentinel.mode}
        self.disconnect_vol_call = [mock.sentinel.device]
        self.detach_call = [mock.sentinel.client, mock.sentinel.attachment_id]
        self.attachment_dict = {'id': mock.sentinel.attachment_id}

    def _sentinel_attach(self):
        attachment_id = self.m.attach(mock.sentinel.client, mock.sentinel.volume_id, mock.sentinel.host, mode=mock.sentinel.mode)
        return attachment_id

    def _sentinel_detach(self, conn):
        self.m.detach(mock.sentinel.client, mock.sentinel.attachment_id, mock.sentinel.volume_id, mock.sentinel.host, conn, mock.sentinel.connection_info, mock.sentinel.device)

    @mock.patch.object(cinder_utils.API, 'attachment_create')
    def test_attach(self, mock_attach_create):
        mock_attach_create.return_value = self.attachment_dict
        attachment = self._sentinel_attach()
        mock_attach_create.assert_called_once_with(*self.attach_call_1, **self.attach_call_2)
        self.assertEqual(mock.sentinel.attachment_id, attachment['id'])

    @mock.patch.object(cinder_utils.API, 'attachment_delete')
    def test_detach_without_attach(self, mock_attach_delete):
        ex = exceptions.BackendException
        conn = mock.MagicMock()
        mock_attach_delete.side_effect = ex()
        self.assertRaises(ex, self._sentinel_detach, conn)
        conn.disconnect_volume.assert_called_once_with(*self.disconnect_vol_call)

    @mock.patch.object(cinder_utils.API, 'attachment_create')
    @mock.patch.object(cinder_utils.API, 'attachment_delete')
    def test_detach_with_attach(self, mock_attach_delete, mock_attach_create):
        conn = mock.MagicMock()
        mock_attach_create.return_value = self.attachment_dict
        attachment = self._sentinel_attach()
        self._sentinel_detach(conn)
        mock_attach_create.assert_called_once_with(*self.attach_call_1, **self.attach_call_2)
        self.assertEqual(mock.sentinel.attachment_id, attachment['id'])
        conn.disconnect_volume.assert_called_once_with(*self.disconnect_vol_call)
        mock_attach_delete.assert_called_once_with(*self.detach_call)

    @mock.patch.object(cinder_utils.API, 'attachment_create')
    def test_attach_fails(self, mock_attach_create):
        mock_attach_create.side_effect = cinder_exception.BadRequest(code=400)
        self.assertRaises(cinder_exception.BadRequest, self.m.attach, mock.sentinel.client, mock.sentinel.volume_id, mock.sentinel.host, mode=mock.sentinel.mode)