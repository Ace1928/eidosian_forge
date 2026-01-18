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
def _init_hyperv_conn(self, host, timeout):

    @_utils.wmi_retry_decorator(error_codes=(w_const.ERROR_SHARING_PAUSED, w_const.EPT_S_NOT_REGISTERED), max_sleep_time=5, max_retry_count=None, timeout=timeout)
    def init():
        try:
            self._conn_cluster = self._get_wmi_conn(self._MS_CLUSTER_NAMESPACE % host)
            self._cluster = self._conn_cluster.MSCluster_Cluster()[0]
            path = self._cluster.path_()
            self._this_node = re.search('\\\\\\\\(.*)\\\\root', path, re.IGNORECASE).group(1)
        except AttributeError:
            raise exceptions.HyperVClusterException(_('Could not initialize cluster wmi connection.'))
    init()