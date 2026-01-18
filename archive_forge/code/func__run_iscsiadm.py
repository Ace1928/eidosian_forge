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
def _run_iscsiadm(self, connection_properties: dict, iscsi_command: tuple[str, ...], **kwargs) -> tuple[str, str]:
    check_exit_code = kwargs.pop('check_exit_code', 0)
    attempts = kwargs.pop('attempts', 1)
    delay_on_retry = kwargs.pop('delay_on_retry', True)
    out, err = self._execute('iscsiadm', '-m', 'node', '-T', connection_properties['target_iqn'], '-p', connection_properties['target_portal'], *iscsi_command, run_as_root=True, root_helper=self._root_helper, check_exit_code=check_exit_code, attempts=attempts, delay_on_retry=delay_on_retry)
    msg = 'iscsiadm %(iscsi_command)s: stdout=%(out)s stderr=%(err)s' % {'iscsi_command': iscsi_command, 'out': out, 'err': err}
    LOG.debug(strutils.mask_password(msg))
    return (out, err)