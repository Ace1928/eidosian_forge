import errno
import logging
import os
import stat
import urllib
import jsonschema
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store import capabilities
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LW
import glance_store.location
from the filesystem store. The users running the services that are
def _get_capacity_info(self, mount_point):
    """Calculates total available space for given mount point.

        :mount_point is path of glance data directory
        """
    stvfs_result = os.statvfs(mount_point)
    total_available_space = stvfs_result.f_bavail * stvfs_result.f_bsize
    return max(0, total_available_space)