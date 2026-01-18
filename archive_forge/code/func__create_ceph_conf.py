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
@classmethod
def _create_ceph_conf(cls, monitor_ips: list[str], monitor_ports: list[str], cluster_name: str, user: str, keyring) -> str:
    monitors = ['%s:%s' % (ip, port) for ip, port in zip(cls._sanitize_mon_hosts(monitor_ips), monitor_ports)]
    mon_hosts = 'mon_host = %s' % ','.join(monitors)
    keyring = cls._check_or_get_keyring_contents(keyring, cluster_name, user)
    try:
        fd, ceph_conf_path = tempfile.mkstemp(prefix='brickrbd_')
        with os.fdopen(fd, 'w') as conf_file:
            conf_file.writelines(['[global]', '\n', mon_hosts, '\n', keyring, '\n'])
        return ceph_conf_path
    except IOError:
        msg = _('Failed to write data to %s.') % ceph_conf_path
        raise exception.BrickException(msg=msg)