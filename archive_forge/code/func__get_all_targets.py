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
def _get_all_targets(self, connection_properties: dict) -> list[tuple[str, str, list]]:
    addressing_mode = connection_properties.get('addressing_mode')
    res = super()._get_all_targets(connection_properties)
    return [(portal, iqn, self._linuxscsi.lun_for_addressing(lun, addressing_mode)) for portal, iqn, lun in res]