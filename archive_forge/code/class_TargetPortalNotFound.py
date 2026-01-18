from __future__ import annotations
import traceback
from typing import Any, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.i18n import _
class TargetPortalNotFound(BrickException):
    message = _('Unable to find target portal %(target_portal)s.')