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
class ClusterGroupStateChangeListenerTestCase(test_base.OsWinBaseTestCase):
    _FAKE_GROUP_NAME = 'fake_group_name'

    @mock.patch.object(clusterutils._ClusterEventListener, '_setup')
    def setUp(self, mock_setup):
        super(ClusterGroupStateChangeListenerTestCase, self).setUp()
        self._listener = clusterutils._ClusterGroupStateChangeListener(mock.sentinel.cluster_handle, self._FAKE_GROUP_NAME)
        self._listener._clusapi_utils = mock.Mock()
        self._clusapi = self._listener._clusapi_utils

    def _get_fake_event(self, **kwargs):
        event = dict(cluster_object_name=self._FAKE_GROUP_NAME.upper(), object_type=mock.sentinel.object_type, filter_flags=mock.sentinel.filter_flags, buff=mock.sentinel.buff, buff_sz=mock.sentinel.buff_sz)
        event.update(**kwargs)
        return event

    def _get_exp_processed_event(self, event, **kwargs):
        preserved_keys = ['cluster_object_name', 'object_type', 'filter_flags', 'notif_key']
        exp_proc_evt = {key: event[key] for key in preserved_keys}
        exp_proc_evt.update(**kwargs)
        return exp_proc_evt

    @mock.patch('ctypes.byref')
    def test_process_event_dropped(self, mock_byref):
        event = self._get_fake_event(cluster_object_name='other_group_name')
        self.assertIsNone(self._listener._process_event(event))
        event = self._get_fake_event(notif_key=2)
        self.assertIsNone(self._listener._process_event(event))
        notif_key = self._listener._NOTIF_KEY_GROUP_COMMON_PROP
        self._clusapi.get_cluster_group_status_info.side_effect = exceptions.ClusterPropertyListEntryNotFound(property_name='fake_prop_name')
        event = self._get_fake_event(notif_key=notif_key)
        self.assertIsNone(self._listener._process_event(event))

    def test_process_state_change_event(self):
        fake_state = constants.CLUSTER_GROUP_ONLINE
        event_buff = ctypes.c_ulong(fake_state)
        notif_key = self._listener._NOTIF_KEY_GROUP_STATE
        event = self._get_fake_event(notif_key=notif_key, buff=ctypes.byref(event_buff), buff_sz=ctypes.sizeof(event_buff))
        exp_proc_evt = self._get_exp_processed_event(event, state=fake_state)
        proc_evt = self._listener._process_event(event)
        self.assertEqual(exp_proc_evt, proc_evt)

    @mock.patch('ctypes.byref')
    def test_process_status_info_change_event(self, mock_byref):
        self._clusapi.get_cluster_group_status_info.return_value = mock.sentinel.status_info
        mock_byref.side_effect = lambda x: ('byref', x)
        notif_key = self._listener._NOTIF_KEY_GROUP_COMMON_PROP
        event = self._get_fake_event(notif_key=notif_key)
        exp_proc_evt = self._get_exp_processed_event(event, status_info=mock.sentinel.status_info)
        proc_evt = self._listener._process_event(event)
        self.assertEqual(exp_proc_evt, proc_evt)
        self._clusapi.get_cluster_group_status_info.assert_called_once_with(mock_byref(mock.sentinel.buff), mock.sentinel.buff_sz)