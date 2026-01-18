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
class _ClusterEventListener(object):
    _notif_keys = {}
    _notif_port_h = None
    _cluster_handle = None
    _running = False
    _stop_on_error = True
    _error_sleep_interval = 2

    def __init__(self, cluster_handle, stop_on_error=True):
        self._cluster_handle = cluster_handle
        self._stop_on_error = stop_on_error
        self._clusapi_utils = _clusapi_utils.ClusApiUtils()
        self._event_queue = queue.Queue()
        self._setup()

    def __enter__(self):
        self._ensure_listener_running()
        return self

    def _get_notif_key_dw(self, notif_key):
        notif_key_dw = self._notif_keys.get(notif_key)
        if notif_key_dw is None:
            notif_key_dw = wintypes.DWORD(notif_key)
            self._notif_keys[notif_key] = notif_key_dw
        return notif_key_dw

    def _add_filter(self, notif_filter, notif_key=0):
        notif_key_dw = self._get_notif_key_dw(notif_key)
        self._notif_port_h = self._clusapi_utils.create_cluster_notify_port_v2(self._cluster_handle, notif_filter, self._notif_port_h, notif_key_dw)

    def _setup_notif_port(self):
        for notif_filter in self._notif_filters_list:
            filter_struct = clusapi_def.NOTIFY_FILTER_AND_TYPE(dwObjectType=notif_filter['object_type'], FilterFlags=notif_filter['filter_flags'])
            notif_key = notif_filter.get('notif_key', 0)
            self._add_filter(filter_struct, notif_key)

    def _setup(self):
        self._setup_notif_port()
        worker = threading.Thread(target=self._listen)
        worker.daemon = True
        self._running = True
        worker.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _signal_stopped(self):
        self._running = False
        self._event_queue.put(None)

    def stop(self):
        self._signal_stopped()
        if self._notif_port_h:
            self._clusapi_utils.close_cluster_notify_port(self._notif_port_h)

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

    def _process_event(self, event):
        return event

    def get(self, timeout=None):
        self._ensure_listener_running()
        event = self._event_queue.get(timeout=timeout)
        self._ensure_listener_running()
        return event

    def _ensure_listener_running(self):
        if not self._running:
            raise exceptions.OSWinException(_('Cluster event listener is not running.'))