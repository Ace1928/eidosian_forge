from __future__ import annotations
import os
import tempfile
import typing
from typing import Any, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rbd as rbd_privsep
from os_brick import utils
def _get_rbd_handle(self, connection_properties: dict[str, Any]) -> linuxrbd.RBDVolumeIOWrapper:
    try:
        user = connection_properties['auth_username']
        pool, volume = connection_properties['name'].split('/')
        cluster_name = connection_properties['cluster_name']
        monitor_ips = connection_properties['hosts']
        monitor_ports = connection_properties['ports']
        keyring = connection_properties.get('keyring')
    except (KeyError, ValueError):
        msg = _('Connect volume failed, malformed connection properties.')
        raise exception.BrickException(msg=msg)
    conf = self._create_ceph_conf(monitor_ips, monitor_ports, str(cluster_name), user, keyring)
    try:
        rbd_client = linuxrbd.RBDClient(user, pool, conffile=conf, rbd_cluster_name=str(cluster_name))
        rbd_volume = linuxrbd.RBDVolume(rbd_client, volume)
        rbd_handle = linuxrbd.RBDVolumeIOWrapper(linuxrbd.RBDImageMetadata(rbd_volume, pool, user, conf))
    except Exception:
        fileutils.delete_if_exists(conf)
        raise
    return rbd_handle