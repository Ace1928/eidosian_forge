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
class _ClusterGroupStateChangeListener(_ClusterEventListener):
    _NOTIF_KEY_GROUP_STATE = 0
    _NOTIF_KEY_GROUP_COMMON_PROP = 1
    _notif_filters_list = [dict(object_type=w_const.CLUSTER_OBJECT_TYPE_GROUP, filter_flags=w_const.CLUSTER_CHANGE_GROUP_STATE_V2, notif_key=_NOTIF_KEY_GROUP_STATE), dict(object_type=w_const.CLUSTER_OBJECT_TYPE_GROUP, filter_flags=w_const.CLUSTER_CHANGE_GROUP_COMMON_PROPERTY_V2, notif_key=_NOTIF_KEY_GROUP_COMMON_PROP)]

    def __init__(self, cluster_handle, group_name=None, **kwargs):
        self._group_name = group_name
        super(_ClusterGroupStateChangeListener, self).__init__(cluster_handle, **kwargs)

    def _process_event(self, event):
        group_name = event['cluster_object_name']
        if self._group_name and self._group_name.lower() != group_name.lower():
            return
        preserved_keys = ['cluster_object_name', 'object_type', 'filter_flags', 'notif_key']
        processed_event = {key: event[key] for key in preserved_keys}
        notif_key = event['notif_key']
        if notif_key == self._NOTIF_KEY_GROUP_STATE:
            if event['buff_sz'] != ctypes.sizeof(wintypes.DWORD):
                raise exceptions.ClusterPropertyRetrieveFailed()
            state_p = ctypes.cast(event['buff'], wintypes.PDWORD)
            state = state_p.contents.value
            processed_event['state'] = state
            return processed_event
        elif notif_key == self._NOTIF_KEY_GROUP_COMMON_PROP:
            try:
                status_info = self._clusapi_utils.get_cluster_group_status_info(ctypes.byref(event['buff']), event['buff_sz'])
                processed_event['status_info'] = status_info
                return processed_event
            except exceptions.ClusterPropertyListEntryNotFound:
                pass