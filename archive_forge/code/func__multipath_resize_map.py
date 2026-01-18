from __future__ import annotations
import glob
import os
import re
import time
from typing import Optional
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import constants
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _multipath_resize_map(self, dm_path):
    cmd = ('multipathd', 'resize', 'map', dm_path)
    out, _err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper)
    if 'fail' in out or 'timeout' in out:
        raise putils.ProcessExecutionError(stdout=out, stderr=_err, exit_code=1, cmd=cmd)
    return out