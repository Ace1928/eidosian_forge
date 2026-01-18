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
def _run_iscsiadm_update_discoverydb(self, connection_properties: dict, iscsi_transport: str='default') -> tuple[str, str]:
    return self._execute('iscsiadm', '-m', 'discoverydb', '-t', 'sendtargets', '-I', iscsi_transport, '-p', connection_properties['target_portal'], '--op', 'update', '-n', 'discovery.sendtargets.auth.authmethod', '-v', connection_properties['discovery_auth_method'], '-n', 'discovery.sendtargets.auth.username', '-v', connection_properties['discovery_auth_username'], '-n', 'discovery.sendtargets.auth.password', '-v', connection_properties['discovery_auth_password'], run_as_root=True, root_helper=self._root_helper)