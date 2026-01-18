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
def _get_login_opts(self, auth_username=None, auth_password=None, auth_type=None, login_flags=0):
    if auth_type is None:
        auth_type = constants.ISCSI_CHAP_AUTH_TYPE if auth_username and auth_password else constants.ISCSI_NO_AUTH_TYPE
    login_opts = iscsi_struct.ISCSI_LOGIN_OPTIONS()
    info_bitmap = 0
    if auth_username:
        login_opts.Username = six.b(auth_username)
        login_opts.UsernameLength = len(auth_username)
        info_bitmap |= w_const.ISCSI_LOGIN_OPTIONS_USERNAME
    if auth_password:
        login_opts.Password = six.b(auth_password)
        login_opts.PasswordLength = len(auth_password)
        info_bitmap |= w_const.ISCSI_LOGIN_OPTIONS_PASSWORD
    login_opts.AuthType = auth_type
    info_bitmap |= w_const.ISCSI_LOGIN_OPTIONS_AUTH_TYPE
    login_opts.InformationSpecified = info_bitmap
    login_opts.LoginFlags = login_flags
    return login_opts