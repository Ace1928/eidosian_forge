from __future__ import annotations
import errno
import functools
import glob
import json
import os.path
import time
from typing import (Callable, Optional, Sequence, Type, Union)  # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _handle_replicated_volume(self, host_device_paths: list[str], conn_props: NVMeOFConnProps) -> str:
    """Assemble the raid from found devices."""
    path_in_raid = False
    for dev_path in host_device_paths:
        path_in_raid = self._is_device_in_raid(dev_path)
        if path_in_raid:
            break
    device_path = RAID_PATH + conn_props.alias
    if path_in_raid:
        self.stop_and_assemble_raid(host_device_paths, device_path, False)
    else:
        paths_found = len(host_device_paths)
        if conn_props.replica_count > paths_found:
            LOG.error('Cannot create MD as %s out of %s legs were found.', paths_found, conn_props.replica_count)
            raise exception.VolumeDeviceNotFound(device=conn_props.alias)
        self.create_raid(host_device_paths, '1', conn_props.alias, conn_props.alias, False)
    return device_path