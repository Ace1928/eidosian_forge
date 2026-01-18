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
def _get_node_startup_values(self, connection_properties):
    out, __ = self._run_iscsiadm_bare(['-m', 'node', '--op', 'show', '-p', connection_properties['target_portal']], check_exit_code=(0, 21)) or ''
    node_values_str = out.strip()
    node_values = node_values_str.split('\n')
    iqn = None
    startup = None
    startup_values = {}
    for node_value in node_values:
        node_keys = node_value.split()
        try:
            if node_keys[0] == 'node.name':
                iqn = node_keys[2]
            elif node_keys[0] == 'node.startup':
                startup = node_keys[2]
            if iqn and startup:
                startup_values[iqn] = startup
                iqn = None
                startup = None
        except IndexError:
            pass
    return startup_values