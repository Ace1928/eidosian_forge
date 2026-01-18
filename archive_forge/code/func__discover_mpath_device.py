from __future__ import annotations
import functools
import glob
import os
import typing
from typing import Optional
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import reflection
from oslo_utils import timeutils
from os_brick import exception
from os_brick import initiator
from os_brick.initiator import host_driver
from os_brick.initiator import initiator_connector
from os_brick.initiator import linuxscsi
from os_brick import utils
def _discover_mpath_device(self, device_wwn: str, connection_properties: dict, device_name: str) -> tuple[str, str]:
    """This method discovers a multipath device.

        Discover a multipath device based on a defined connection_property
        and a device_wwn and return the multipath_id and path of the multipath
        enabled device if there is one.
        """
    path = self._linuxscsi.find_multipath_device_path(device_wwn)
    device_path = None
    multipath_id = None
    if path is None:
        device_realpath = os.path.realpath(device_name)
        mpath_info = self._linuxscsi.find_multipath_device(device_realpath)
        if mpath_info:
            device_path = mpath_info['device']
            multipath_id = device_wwn
        else:
            device_path = device_name
            LOG.debug('Unable to find multipath device name for volume. Using path %(device)s for volume.', {'device': device_path})
    else:
        device_path = path
        multipath_id = device_wwn
    if connection_properties.get('access_mode', '') != 'ro':
        try:
            self._linuxscsi.wait_for_rw(device_wwn, device_path)
        except exception.BlockDeviceReadOnly:
            LOG.warning('Block device %s is still read-only. Continuing anyway.', device_path)
    device_path = typing.cast(str, device_path)
    multipath_id = typing.cast(str, multipath_id)
    return (device_path, multipath_id)