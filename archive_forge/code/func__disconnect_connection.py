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
def _disconnect_connection(self, connection_properties: dict, connections: Iterable, force: bool, exc) -> None:
    LOG.debug('Disconnecting from: %s', connections)
    props = connection_properties.copy()
    for ip, iqn in connections:
        props['target_portal'] = ip
        props['target_iqn'] = iqn
        with exc.context(force, 'Disconnect from %s %s failed', ip, iqn):
            self._disconnect_from_iscsi_portal(props)