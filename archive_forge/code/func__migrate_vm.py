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
def _migrate_vm(self, vm_name, new_host, migration_type, exp_state_after_migr, timeout):
    syntax = w_const.CLUSPROP_SYNTAX_LIST_VALUE_DWORD
    migr_type = wintypes.DWORD(migration_type)
    prop_entries = [self._clusapi_utils.get_property_list_entry(w_const.CLUS_RESTYPE_NAME_VM, syntax, migr_type), self._clusapi_utils.get_property_list_entry(w_const.CLUS_RESTYPE_NAME_VM_CONFIG, syntax, migr_type)]
    prop_list = self._clusapi_utils.get_property_list(prop_entries)
    flags = w_const.CLUSAPI_GROUP_MOVE_RETURN_TO_SOURCE_NODE_ON_ERROR | w_const.CLUSAPI_GROUP_MOVE_QUEUE_ENABLED | w_const.CLUSAPI_GROUP_MOVE_HIGH_PRIORITY_START
    with self._cmgr.open_cluster() as cluster_handle, self._cmgr.open_cluster_group(vm_name, cluster_handle=cluster_handle) as group_handle, self._cmgr.open_cluster_node(new_host, cluster_handle=cluster_handle) as dest_node_handle, _ClusterGroupStateChangeListener(cluster_handle, vm_name) as listener:
        self._clusapi_utils.move_cluster_group(group_handle, dest_node_handle, flags, prop_list)
        try:
            self._wait_for_cluster_group_migration(listener, vm_name, group_handle, exp_state_after_migr, timeout)
        except exceptions.ClusterGroupMigrationTimeOut:
            with excutils.save_and_reraise_exception() as ctxt:
                self._cancel_cluster_group_migration(listener, vm_name, group_handle, exp_state_after_migr, timeout)
                try:
                    self._validate_migration(group_handle, vm_name, exp_state_after_migr, new_host)
                    LOG.warning('Cluster group migration completed successfuly after cancel attempt. Suppressing timeout exception.')
                    ctxt.reraise = False
                except exceptions.ClusterGroupMigrationFailed:
                    pass
        else:
            self._validate_migration(group_handle, vm_name, exp_state_after_migr, new_host)