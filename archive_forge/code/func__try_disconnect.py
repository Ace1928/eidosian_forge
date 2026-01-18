from __future__ import annotations
import errno
import functools
import glob
import json
import os.path
import time
from typing import (Callable, Optional, Sequence, Type, Union)  # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _try_disconnect(self, portal: Portal) -> None:
    """Disconnect a specific subsystem if it's safe.

        Only disconnect if it has no namespaces left or has only one left and
        it is from this connection.
        """
    LOG.debug('Checking if %s can be disconnected', portal)
    if portal.can_disconnect():
        self._execute('nvme', 'disconnect', '-d', '/dev/' + portal.controller, root_helper=self._root_helper, run_as_root=True)