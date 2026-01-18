from __future__ import annotations
import math
import os
import re
from typing import Any, Callable, Optional  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from os_brick import exception
from os_brick import executor
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _calculate_thin_pool_size(self) -> list[str]:
    """Calculates the correct size for a thin pool.

        Some amount of free space must remain in the volume group for
        metadata for the contained logical volumes.  The exact amount depends
        on how much volume sharing you expect.

        :returns: An lvcreate-ready string for the number of calculated bytes.
        """
    self.update_volume_group_info()
    return ['-l', '100%FREE']