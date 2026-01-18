import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def _drive_to_boot_source(self, drive_path):
    is_physical = 'root\\virtualization\\v2:Msvm_DiskDrive'.lower() in drive_path.lower()
    drive = self._get_mounted_disk_resource_from_path(drive_path, is_physical=is_physical)
    rasd_path = drive.path_() if is_physical else drive.Parent
    bssd = self._conn.Msvm_LogicalIdentity(SystemElement=rasd_path)[0].SameElement
    return bssd.path_()