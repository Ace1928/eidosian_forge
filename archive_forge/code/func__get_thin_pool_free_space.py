from __future__ import annotations
import math
import os
import re
from typing import Any, Callable, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _get_thin_pool_free_space(self, vg_name: str, thin_pool_name: str) -> float:
    """Returns available thin pool free space.

        :param vg_name: the vg where the pool is placed
        :param thin_pool_name: the thin pool to gather info for
        :returns: Free space in GB (float), calculated using data_percent

        """
    cmd = LVM.LVM_CMD_PREFIX + ['lvs', '--noheadings', '--unit=g', '-o', 'size,data_percent', '--separator', ':', '--nosuffix']
    cmd.append('/dev/%s/%s' % (vg_name, thin_pool_name))
    free_space = 0.0
    try:
        out, err = self._run_lvm_command(cmd)
        if out is not None:
            out = out.strip()
            data = out.split(':')
            pool_size = float(data[0])
            data_percent = float(data[1])
            consumed_space = pool_size / 100 * data_percent
            free_space = pool_size - consumed_space
            free_space = round(free_space, 2)
    except putils.ProcessExecutionError as err:
        LOG.exception('Error querying thin pool about data_percent')
        LOG.error('Cmd     :%s', err.cmd)
        LOG.error('StdOut  :%s', err.stdout)
        LOG.error('StdErr  :%s', err.stderr)
    return free_space