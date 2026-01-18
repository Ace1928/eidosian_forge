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
@utils.retry(retry=utils.retry_if_exit_code, retry_param=139, interval=0.5, backoff_rate=0.5)
def _run_lvm_command(self, cmd_arg_list: list[str], root_helper: Optional[str]=None, run_as_root: bool=True) -> tuple[str, str]:
    """Run LVM commands with a retry on code 139 to work around LVM bugs.

        Refer to LP bug 1901783, LP bug 1932188.
        """
    if not root_helper:
        root_helper = self._root_helper
    out, err = self._execute(*cmd_arg_list, root_helper=root_helper, run_as_root=run_as_root)
    return (out, err)