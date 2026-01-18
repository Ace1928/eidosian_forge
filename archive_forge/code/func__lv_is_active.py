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
def _lv_is_active(self, name: str) -> bool:
    cmd = LVM.LVM_CMD_PREFIX + ['lvdisplay', '--noheading', '-C', '-o', 'Attr', '%s/%s' % (self.vg_name, name)]
    out, _err = self._run_lvm_command(cmd)
    if out:
        out = out.strip()
        if out[4] == 'a':
            return True
    return False