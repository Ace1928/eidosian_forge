import ctypes
import re
import sys
import threading
import time
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import excutils
from six.moves import queue
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def _setup_notif_port(self):
    for notif_filter in self._notif_filters_list:
        filter_struct = clusapi_def.NOTIFY_FILTER_AND_TYPE(dwObjectType=notif_filter['object_type'], FilterFlags=notif_filter['filter_flags'])
        notif_key = notif_filter.get('notif_key', 0)
        self._add_filter(filter_struct, notif_key)