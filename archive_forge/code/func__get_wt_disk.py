from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import hostutils
from os_win.utils import pathutils
from os_win.utils import win32utils
def _get_wt_disk(self, description, fail_if_not_found=True):
    wt_disks = self._conn_wmi.WT_Disk(Description=description)
    if wt_disks:
        return wt_disks[0]
    elif fail_if_not_found:
        err_msg = _('Could not find WT Disk: %s')
        raise exceptions.ISCSITargetException(err_msg % description)