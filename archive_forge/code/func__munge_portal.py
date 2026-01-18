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
def _munge_portal(self, target: tuple[str, str, Union[list, str]]) -> tuple[str, str, str]:
    """Remove brackets from portal.

        In case IPv6 address was used the udev path should not contain any
        brackets. Udev code specifically forbids that.
        """
    portal, iqn, lun = target
    return (portal.replace('[', '').replace(']', ''), iqn, self._linuxscsi.process_lun_id(lun))