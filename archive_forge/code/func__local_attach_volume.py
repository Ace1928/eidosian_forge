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
def _local_attach_volume(self, connection_properties: dict[str, Any]) -> dict[str, Union[str, linuxrbd.RBDVolumeIOWrapper]]:
    try:
        self._execute('which', 'rbd')
    except putils.ProcessExecutionError:
        msg = _('ceph-common package is not installed.')
        LOG.error(msg)
        raise exception.BrickException(message=msg)
    pool, volume = connection_properties['name'].split('/')
    rbd_dev_path = self.get_rbd_device_name(pool, volume)
    conf = self.create_non_openstack_config(connection_properties)
    try:
        if not os.path.islink(rbd_dev_path) or not os.path.exists(os.path.realpath(rbd_dev_path)):
            cmd = ['rbd', 'map', volume, '--pool', pool]
            cmd += self._get_rbd_args(connection_properties, conf)
            self._execute(*cmd, root_helper=self._root_helper, run_as_root=True)
        else:
            LOG.debug('Volume %(vol)s is already mapped to local device %(dev)s', {'vol': volume, 'dev': os.path.realpath(rbd_dev_path)})
        if not os.path.islink(rbd_dev_path) or not os.path.exists(os.path.realpath(rbd_dev_path)):
            LOG.warning('Volume %(vol)s has not been mapped to local device %(dev)s; is the udev daemon running and are the ceph-renamer udev rules configured? See bug #1884114 for more information.', {'vol': volume, 'dev': rbd_dev_path})
    except Exception:
        with excutils.save_and_reraise_exception():
            if conf:
                rbd_privsep.delete_if_exists(conf)
    res: dict[str, Union[str, linuxrbd.RBDVolumeIOWrapper]]
    res = {'path': rbd_dev_path, 'type': 'block'}
    if conf:
        res['conf'] = conf
    return res