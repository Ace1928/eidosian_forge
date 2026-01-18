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
def _get_host_devices(self, possible_devs: list) -> list:
    """Compute the device paths on the system with an id, wwn, and lun

        :param possible_devs: list of (platform, pci_id, wwn, lun) tuples
        :return: list of device paths on the system based on the possible_devs
        """
    host_devices = []
    for platform, pci_num, target_wwn, lun in possible_devs:
        host_device = '/dev/disk/by-path/%spci-%s-fc-%s-lun-%s' % (platform + '-' if platform else '', pci_num, target_wwn, self._linuxscsi.process_lun_id(lun))
        host_devices.append(host_device)
    return host_devices