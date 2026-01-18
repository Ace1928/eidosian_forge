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
@staticmethod
def _is_native_multipath_supported():
    try:
        with open('/sys/module/nvme_core/parameters/multipath', 'rt') as f:
            return f.read().strip() == 'Y'
    except Exception:
        LOG.warning('Could not find nvme_core/parameters/multipath')
    return False