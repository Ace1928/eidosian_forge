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
def _new_session_required(self, target_iqn, portal_addr, portal_port, initiator_name, mpio_enabled):
    login_required = False
    sessions = self._get_iscsi_target_sessions(target_iqn)
    if not sessions:
        login_required = True
    elif mpio_enabled:
        login_required = not self._session_on_path_exists(sessions, portal_addr, portal_port, initiator_name)
    return login_required