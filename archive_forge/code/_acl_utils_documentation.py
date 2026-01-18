import ctypes
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
Merge new ACEs into an existing ACL, returning a new ACL.