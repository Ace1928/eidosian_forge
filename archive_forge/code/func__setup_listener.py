import ctypes
from unittest import mock
import ddt
from six.moves import queue
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import clusterutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def _setup_listener(self, stop_on_error=True):
    self._listener = clusterutils._ClusterEventListener(mock.sentinel.cluster_handle, stop_on_error=stop_on_error)
    self._listener._running = True
    self._listener._clusapi_utils = mock.Mock()
    self._clusapi = self._listener._clusapi_utils