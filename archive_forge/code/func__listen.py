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
def _listen(self):
    while self._running:
        try:
            event = _utils.avoid_blocking_call(self._clusapi_utils.get_cluster_notify_v2, self._notif_port_h, timeout_ms=-1)
            processed_event = self._process_event(event)
            if processed_event:
                self._event_queue.put(processed_event)
        except Exception:
            if self._running:
                LOG.exception('Unexpected exception in event listener loop.')
                if self._stop_on_error:
                    LOG.warning('The cluster event listener will now close.')
                    self._signal_stopped()
                else:
                    time.sleep(self._error_sleep_interval)