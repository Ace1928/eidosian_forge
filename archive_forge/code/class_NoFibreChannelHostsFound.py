from __future__ import annotations
import traceback
from typing import Any, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.i18n import _
class NoFibreChannelHostsFound(BrickException):
    message = _('We are unable to locate any Fibre Channel devices.')