import contextlib
import errno
import importlib
import logging
import math
import os
import shlex
import socket
import time
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import exceptions as keystone_exc
from keystoneauth1 import identity as ksa_identity
from keystoneauth1 import session as ksa_session
from keystoneauth1 import token_endpoint as ksa_token_endpoint
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_utils import strutils
from oslo_utils import units
from glance_store._drivers.cinder import base
from glance_store import capabilities
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI, _LW
import glance_store.location
from the service catalog, and current context's user and project are used.
@contextlib.contextmanager
def _open_cinder_volume(self, client, volume, mode):
    attach_mode = 'rw' if mode == 'wb' else 'ro'
    device = None
    root_helper = self.get_root_helper()
    priv_context.init(root_helper=shlex.split(root_helper))
    host = socket.gethostname()
    my_ip = self._get_host_ip(host)
    use_multipath = self.store_conf.cinder_use_multipath
    enforce_multipath = self.store_conf.cinder_enforce_multipath
    volume_id = volume.id
    connector_prop = connector.get_connector_properties(root_helper, my_ip, use_multipath, enforce_multipath, host=host)
    if volume.multiattach:
        attachment = attachment_state_manager.attach(client, volume_id, host, mode=attach_mode)
    else:
        attachment = self.volume_api.attachment_create(client, volume_id, mode=attach_mode)
    LOG.debug('Attachment %(attachment_id)s created successfully.', {'attachment_id': attachment['id']})
    volume = volume.manager.get(volume_id)
    attachment_id = attachment['id']
    connection_info = None
    try:
        attachment = self.volume_api.attachment_update(client, attachment_id, connector_prop, mountpoint='glance_store')
        LOG.debug('Attachment %(attachment_id)s updated successfully with connection info %(conn_info)s', {'attachment_id': attachment_id, 'conn_info': strutils.mask_dict_password(attachment.connection_info)})
        connection_info = attachment.connection_info
        conn = base.factory(connection_info['driver_volume_type'], volume=volume, connection_info=connection_info, root_helper=root_helper, use_multipath=use_multipath, mountpoint_base=self.store_conf.cinder_mount_point_base, attachment_obj=attachment, client=client)
        device = conn.connect_volume(volume)
        self.volume_api.attachment_complete(client, attachment_id)
        LOG.debug('Attachment %(attachment_id)s completed successfully.', {'attachment_id': attachment_id})
        self.volume_connector_map[volume.id] = conn
        if connection_info['driver_volume_type'] == 'rbd' and (not conn.conn.do_local_attach):
            yield device['path']
        else:
            with self.temporary_chown(device['path']), open(device['path'], mode) as f:
                yield conn.yield_path(volume, f)
    except Exception:
        LOG.exception(_LE('Exception while accessing to cinder volume %(volume_id)s.'), {'volume_id': volume.id})
        raise
    finally:
        if device:
            try:
                if volume.multiattach:
                    attachment_state_manager.detach(client, attachment_id, volume_id, host, conn, connection_info, device)
                else:
                    conn.disconnect_volume(device)
                    if self.volume_connector_map.get(volume.id):
                        del self.volume_connector_map[volume.id]
            except Exception:
                LOG.exception(_LE('Failed to disconnect volume %(volume_id)s.'), {'volume_id': volume.id})
        if not volume.multiattach:
            self.volume_api.attachment_delete(client, attachment_id)