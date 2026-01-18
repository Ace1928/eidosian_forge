import ctypes
import functools
import inspect
import socket
import time
from oslo_log import log as logging
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
def _remove_target_persistent_logins(self, target_iqn):
    persistent_logins = self._get_iscsi_persistent_logins()
    for persistent_login in persistent_logins:
        if persistent_login.TargetName == target_iqn:
            LOG.debug('Removing iSCSI target persistent login: %(target_iqn)s', dict(target_iqn=target_iqn))
            self._remove_persistent_login(persistent_login)