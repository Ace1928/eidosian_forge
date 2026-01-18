from __future__ import annotations
import os
import typing
from typing import Any, Optional  # noqa: H301
from oslo_log import log as logging
from oslo_service import loopingcall
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator import linuxfc
from os_brick import utils
def _remove_devices(self, connection_properties: dict, devices: list, device_info: dict, force: bool, exc) -> None:
    path_used = utils.get_dev_path(connection_properties, device_info)
    was_symlink = path_used.count(os.sep) > 2
    was_multipath = '/pci-' not in path_used and was_symlink
    for device in devices:
        with exc.context(force, 'Removing device %s failed', device):
            device_path = device['device']
            flush = self._linuxscsi.requires_flush(device_path, path_used, was_multipath)
            self._linuxscsi.remove_scsi_device(device_path, flush=flush)