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
def _get_fake_event(self, **kwargs):
    event = dict(cluster_object_name=self._FAKE_GROUP_NAME.upper(), object_type=mock.sentinel.object_type, filter_flags=mock.sentinel.filter_flags, buff=mock.sentinel.buff, buff_sz=mock.sentinel.buff_sz)
    event.update(**kwargs)
    return event