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
def _create_vg(self, pv_list: list[str]) -> None:
    cmd = ['vgcreate', self.vg_name, ','.join(pv_list)]
    self._execute(*cmd, root_helper=self._root_helper, run_as_root=True)