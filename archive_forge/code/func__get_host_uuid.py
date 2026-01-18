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
def _get_host_uuid(self) -> Optional[str]:
    """Get the UUID of the first mounted filesystem."""
    cmd = ('findmnt', '-v', '/', '-n', '-o', 'SOURCE')
    try:
        lines, err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper)
        blkid_cmd = ('blkid', lines.split('\n')[0], '-s', 'UUID', '-o', 'value')
        lines, _err = self._execute(*blkid_cmd, run_as_root=True, root_helper=self._root_helper)
        return lines.split('\n')[0]
    except putils.ProcessExecutionError as e:
        LOG.warning('Process execution error in _get_host_uuid: %s', e)
        return None