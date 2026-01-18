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
@staticmethod
def _check_valid_device(rbd_handle: 'io.BufferedReader') -> bool:
    original_offset = rbd_handle.tell()
    try:
        rbd_handle.read(4096)
    except Exception as e:
        LOG.error('Failed to access RBD device handle: %(error)s', {'error': e})
        return False
    finally:
        rbd_handle.seek(original_offset, 0)
    return True