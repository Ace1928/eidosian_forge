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
def _try_disconnect_all(self, conn_props: NVMeOFConnProps, exc: Optional[exception.ExceptionChainer]=None) -> None:
    """Disconnect all subsystems that are not being used.

        Only sees if it has to disconnect this connection properties portals,
        leaves other alone.

        Since this is unrelated to the flushing of the devices failures will be
        logged but they won't be raised.
        """
    if exc is None:
        exc = exception.ExceptionChainer()
    for target in conn_props.targets:
        target.set_portals_controllers()
        for portal in target.portals:
            with exc.context(True, 'Failed to disconnect %s', portal):
                self._try_disconnect(portal)