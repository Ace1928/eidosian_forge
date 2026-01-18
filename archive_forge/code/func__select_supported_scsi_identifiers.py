import ctypes
import os
import re
import threading
from collections.abc import Iterable
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import pathutils
from os_win.utils import win32utils
from os_win.utils.winapi import libs as w_lib
def _select_supported_scsi_identifiers(self, identifiers):
    selected_identifiers = []
    for id_type in constants.SUPPORTED_SCSI_UID_FORMATS:
        for identifier in identifiers:
            if identifier['type'] == id_type:
                selected_identifiers.append(identifier)
    return selected_identifiers