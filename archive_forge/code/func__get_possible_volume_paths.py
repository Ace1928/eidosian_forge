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
def _get_possible_volume_paths(self, connection_properties: dict, hbas) -> list[str]:
    targets = connection_properties['targets']
    addressing_mode = connection_properties.get('addressing_mode')
    possible_devs = self._get_possible_devices(hbas, targets, addressing_mode)
    host_paths = self._get_host_devices(possible_devs)
    return host_paths