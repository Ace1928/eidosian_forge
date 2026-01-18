import collections
import contextlib
import logging
import socket
import threading
from oslo_config import cfg
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
class _AttachmentState(object):
    """A data structure recording all managed attachments. _AttachmentState
    ensures that the glance node only attempts to a single multiattach volume
    in use by multiple attachments once, and that it is not disconnected until
    it is no longer in use by any attachments.

    Callers should not create a _AttachmentState directly, but should obtain
    it via:

      with attachment.get_manager().get_state() as state:
        state.attach(...)

    _AttachmentState manages concurrency itself. Independent callers do not
    need to consider interactions between multiple _AttachmentState calls when
    designing their own locking.
    """

    class _Attachment(object):

        def __init__(self):
            self.lock = threading.Lock()
            self.attachments = set()

        def add_attachment(self, attachment_id, host):
            self.attachments.add((attachment_id, host))

        def remove_attachment(self, attachment_id, host):
            self.attachments.remove((attachment_id, host))

        def in_use(self):
            return len(self.attachments) > 0

    def __init__(self):
        """Initialise _AttachmentState"""
        self.volumes = collections.defaultdict(self._Attachment)
        self.volume_api = cinder_utils.API()

    @contextlib.contextmanager
    def _get_locked(self, volume):
        """Get a locked attachment object

        :param mountpoint: The path of the volume whose attachment we should
                           return.
        :rtype: _AttachmentState._Attachment
        """
        while True:
            vol = self.volumes[volume]
            with vol.lock:
                if self.volumes[volume] is vol:
                    yield vol
                    break

    def attach(self, client, volume_id, host, mode=None):
        """Ensure a volume is available for an attachment and create an
        attachment

        :param client: Cinderclient object
        :param volume_id: ID of the volume to attach
        :param host: The host the volume will be attached to
        :param mode: The attachment mode
        """
        LOG.debug('_AttachmentState.attach(volume_id=%(volume_id)s, host=%(host)s, mode=%(mode)s)', {'volume_id': volume_id, 'host': host, 'mode': mode})
        with self._get_locked(volume_id) as vol_attachment:
            try:
                attachment = self.volume_api.attachment_create(client, volume_id, mode=mode)
            except Exception:
                LOG.exception(_LE('Error attaching volume %(volume_id)s'), {'volume_id': volume_id})
                del self.volumes[volume_id]
                raise
            vol_attachment.add_attachment(attachment['id'], host)
        LOG.debug('_AttachmentState.attach for volume_id=%(volume_id)s and attachment_id=%(attachment_id)s completed successfully', {'volume_id': volume_id, 'attachment_id': attachment['id']})
        return attachment

    def detach(self, client, attachment_id, volume_id, host, conn, connection_info, device):
        """Delete the attachment no longer in use, and disconnect volume
        if necessary.

        :param client: Cinderclient object
        :param attachment_id: ID of the attachment between volume and host
        :param volume_id: ID of the volume to attach
        :param host: The host the volume was attached to
        :param conn: connector object
        :param connection_info: connection information of the volume we are
                                detaching
        :device: device used to write image

        """
        LOG.debug('_AttachmentState.detach(vol_id=%(volume_id)s, attachment_id=%(attachment_id)s)', {'volume_id': volume_id, 'attachment_id': attachment_id})
        with self._get_locked(volume_id) as vol_attachment:
            try:
                vol_attachment.remove_attachment(attachment_id, host)
            except KeyError:
                LOG.warning(_LW("Request to remove attachment (%(volume_id)s, %(host)s) but we don't think it's in use."), {'volume_id': volume_id, 'host': host})
            if not vol_attachment.in_use():
                conn.disconnect_volume(device)
                del self.volumes[volume_id]
            self.volume_api.attachment_delete(client, attachment_id)
            LOG.debug('_AttachmentState.detach for volume %(volume_id)s and attachment_id=%(attachment_id)s completed successfully', {'volume_id': volume_id, 'attachment_id': attachment_id})