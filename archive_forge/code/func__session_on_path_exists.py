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
def _session_on_path_exists(self, target_sessions, portal_addr, portal_port, initiator_name):
    for session in target_sessions:
        connections = session.Connections[:session.ConnectionCount]
        uses_requested_initiator = False
        if initiator_name:
            devices = self._get_iscsi_session_devices(session.SessionId)
            for device in devices:
                if device.InitiatorName == initiator_name:
                    uses_requested_initiator = True
                    break
        else:
            uses_requested_initiator = True
        for conn in connections:
            is_requested_path = uses_requested_initiator and conn.TargetAddress == portal_addr and (conn.TargetSocket == portal_port)
            if is_requested_path:
                return True
    return False