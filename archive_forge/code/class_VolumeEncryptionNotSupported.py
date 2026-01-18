from __future__ import annotations
import traceback
from typing import Any, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.i18n import _
class VolumeEncryptionNotSupported(Invalid):
    message = _('Volume encryption is not supported for %(volume_type)s volume %(volume_id)s.')