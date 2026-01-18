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
def _ensure_wt_provider_available(self):
    try:
        self._conn_wmi.WT_Portal
    except AttributeError:
        err_msg = _('The Windows iSCSI target provider is not available.')
        raise exceptions.ISCSITargetException(err_msg)