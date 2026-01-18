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
def _get_vm_groups(self):
    for r in self.cluster_enum(w_const.CLUSTER_ENUM_GROUP):
        group_type = self.get_cluster_group_type(r['name'])
        if group_type == w_const.ClusGroupTypeVirtualMachine:
            yield r