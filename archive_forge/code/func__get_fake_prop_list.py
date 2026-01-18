import ctypes
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def _get_fake_prop_list(self):
    syntax = w_const.CLUSPROP_SYNTAX_LIST_VALUE_DWORD
    migr_type = wintypes.DWORD(self._LIVE_MIGRATION_TYPE)
    prop_entries = [self._clusapi_utils.get_property_list_entry(w_const.CLUS_RESTYPE_NAME_VM, syntax, migr_type), self._clusapi_utils.get_property_list_entry(w_const.CLUS_RESTYPE_NAME_VM_CONFIG, syntax, migr_type), self._clusapi_utils.get_property_list_entry(w_const.CLUSREG_NAME_GRP_STATUS_INFORMATION, w_const.CLUSPROP_SYNTAX_LIST_VALUE_ULARGE_INTEGER, ctypes.c_ulonglong(w_const.CLUSGRP_STATUS_WAITING_IN_QUEUE_FOR_MOVE)), self._clusapi_utils.get_property_list_entry(w_const.CLUSREG_NAME_GRP_TYPE, w_const.CLUSPROP_SYNTAX_LIST_VALUE_DWORD, ctypes.c_ulong(w_const.ClusGroupTypeVirtualMachine))]
    prop_list = self._clusapi_utils.get_property_list(prop_entries)
    return prop_list