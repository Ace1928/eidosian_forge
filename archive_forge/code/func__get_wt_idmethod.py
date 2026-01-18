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
def _get_wt_idmethod(self, initiator, target_name):
    wt_idmethod = self._conn_wmi.WT_IDMethod(HostName=target_name, Value=initiator)
    if wt_idmethod:
        return wt_idmethod[0]