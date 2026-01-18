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
def _cancel_cluster_group_migration(self, event_listener, group_name, group_handle, expected_state, timeout=None):
    LOG.info("Canceling cluster group '%s' migration", group_name)
    try:
        cancel_finished = self._clusapi_utils.cancel_cluster_group_operation(group_handle)
    except exceptions.Win32Exception as ex:
        group_state_info = self._get_cluster_group_state(group_handle)
        migration_pending = self._is_migration_pending(group_state_info['state'], group_state_info['status_info'], expected_state)
        if ex.error_code == w_const.ERROR_INVALID_STATE and (not migration_pending):
            LOG.debug('Ignoring group migration cancel error. No migration is pending.')
            cancel_finished = True
        else:
            raise
    if not cancel_finished:
        LOG.debug('Waiting for group migration to be canceled.')
        try:
            self._wait_for_cluster_group_migration(event_listener, group_name, group_handle, expected_state, timeout=timeout)
        except Exception:
            LOG.exception('Failed to cancel cluster group migration.')
            raise exceptions.JobTerminateFailed()
    LOG.info('Cluster group migration canceled.')