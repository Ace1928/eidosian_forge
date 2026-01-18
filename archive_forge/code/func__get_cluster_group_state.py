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
def _get_cluster_group_state(self, group_handle):
    state_info = self._clusapi_utils.get_cluster_group_state(group_handle)
    buff, buff_sz = self._clusapi_utils.cluster_group_control(group_handle, w_const.CLUSCTL_GROUP_GET_RO_COMMON_PROPERTIES)
    status_info = self._clusapi_utils.get_cluster_group_status_info(ctypes.byref(buff), buff_sz)
    state_info['status_info'] = status_info
    return state_info