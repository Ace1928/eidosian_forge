from __future__ import annotations
from collections import defaultdict
import copy
import glob
import os
import re
import time
from typing import Any, Iterable, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import strutils
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator import utils as initiator_utils
from os_brick import utils
def _get_iscsi_nodes(self) -> list[tuple]:
    """Get iSCSI node information (portal, iqn) as a list of tuples.

        Uses iscsiadm -m node and from a command output like
            192.168.121.250:3260,1 iqn.2010-10.org.openstack:volume

        This method will drop the tpgt and return a list like this:
            [('192.168.121.250:3260', 'iqn.2010-10.org.openstack:volume')]
        """
    out, err = self._execute('iscsiadm', '-m', 'node', run_as_root=True, root_helper=self._root_helper, check_exit_code=False)
    if err:
        LOG.warning("Couldn't find iSCSI nodes because iscsiadm err: %s", err)
        return []
    lines: list[tuple] = []
    for line in out.splitlines():
        if line:
            info = line.split()
            try:
                lines.append((info[0].split(',')[0], info[1]))
            except IndexError:
                pass
    return lines