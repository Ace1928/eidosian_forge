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
def _get_iscsi_sessions_full(self) -> list[tuple[str, str, str, str, str]]:
    """Get iSCSI session information as a list of tuples.

        Uses iscsiadm -m session and from a command output like
            tcp: [1] 192.168.121.250:3260,1 iqn.2010-10.org.openstack:
            volume- (non-flash)

        This method will drop the node type and return a list like this:
            [('tcp:', '1', '192.168.121.250:3260', '1',
              'iqn.2010-10.org.openstack:volume-')]
        """
    out, err = self._run_iscsi_session()
    if err:
        LOG.warning('iscsiadm stderr output when getting sessions: %s', err)
    lines: list[tuple[str, str, str, str, str]] = []
    for line in out.splitlines():
        if line:
            info = line.split()
            sid = info[1][1:-1]
            portal, tpgt = info[2].split(',')
            lines.append((info[0], sid, portal, tpgt, info[3]))
    return lines