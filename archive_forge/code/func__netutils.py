import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
@property
def _netutils(self):
    if not self._netutils_prop:
        from os_win import utilsfactory
        self._netutils_prop = utilsfactory.get_networkutils()
    return self._netutils_prop