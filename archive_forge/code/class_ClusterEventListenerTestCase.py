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
class ClusterEventListenerTestCase(test_base.OsWinBaseTestCase):

    @mock.patch.object(clusterutils._ClusterEventListener, '_setup')
    def setUp(self, mock_setup):
        super(ClusterEventListenerTestCase, self).setUp()
        self._setup_listener()

    def _setup_listener(self, stop_on_error=True):
        self._listener = clusterutils._ClusterEventListener(mock.sentinel.cluster_handle, stop_on_error=stop_on_error)
        self._listener._running = True
        self._listener._clusapi_utils = mock.Mock()
        self._clusapi = self._listener._clusapi_utils

    def test_get_notif_key_dw(self):
        fake_notif_key = 1
        notif_key_dw = self._listener._get_notif_key_dw(fake_notif_key)
        self.assertIsInstance(notif_key_dw, ctypes.c_ulong)
        self.assertEqual(fake_notif_key, notif_key_dw.value)
        self.assertEqual(notif_key_dw, self._listener._get_notif_key_dw(fake_notif_key))

    @mock.patch.object(clusterutils._ClusterEventListener, '_get_notif_key_dw')
    def test_add_filter(self, mock_get_notif_key):
        mock_get_notif_key.side_effect = (mock.sentinel.notif_key_dw, mock.sentinel.notif_key_dw_2)
        self._clusapi.create_cluster_notify_port_v2.return_value = mock.sentinel.notif_port_h
        self._listener._add_filter(mock.sentinel.filter, mock.sentinel.notif_key)
        self._listener._add_filter(mock.sentinel.filter_2, mock.sentinel.notif_key_2)
        self.assertEqual(mock.sentinel.notif_port_h, self._listener._notif_port_h)
        mock_get_notif_key.assert_has_calls([mock.call(mock.sentinel.notif_key), mock.call(mock.sentinel.notif_key_2)])
        self._clusapi.create_cluster_notify_port_v2.assert_has_calls([mock.call(mock.sentinel.cluster_handle, mock.sentinel.filter, None, mock.sentinel.notif_key_dw), mock.call(mock.sentinel.cluster_handle, mock.sentinel.filter_2, mock.sentinel.notif_port_h, mock.sentinel.notif_key_dw_2)])

    @mock.patch.object(clusterutils._ClusterEventListener, '_add_filter')
    @mock.patch.object(clusapi_def, 'NOTIFY_FILTER_AND_TYPE')
    def test_setup_notif_port(self, mock_filter_struct_cls, mock_add_filter):
        notif_filter = dict(object_type=mock.sentinel.object_type, filter_flags=mock.sentinel.filter_flags, notif_key=mock.sentinel.notif_key)
        self._listener._notif_filters_list = [notif_filter]
        self._listener._setup_notif_port()
        mock_filter_struct_cls.assert_called_once_with(dwObjectType=mock.sentinel.object_type, FilterFlags=mock.sentinel.filter_flags)
        mock_add_filter.assert_called_once_with(mock_filter_struct_cls.return_value, mock.sentinel.notif_key)

    def test_signal_stopped(self):
        self._listener._signal_stopped()
        self.assertFalse(self._listener._running)
        self.assertIsNone(self._listener._event_queue.get(block=False))

    @mock.patch.object(clusterutils._ClusterEventListener, '_signal_stopped')
    def test_stop(self, mock_signal_stopped):
        self._listener._notif_port_h = mock.sentinel.notif_port_h
        self._listener.stop()
        mock_signal_stopped.assert_called_once_with()
        self._clusapi.close_cluster_notify_port.assert_called_once_with(mock.sentinel.notif_port_h)

    @mock.patch.object(clusterutils._ClusterEventListener, '_process_event')
    def test_listen(self, mock_process_event):
        events = [mock.sentinel.ignored_event, mock.sentinel.retrieved_event]
        self._clusapi.get_cluster_notify_v2.side_effect = events
        self._listener._notif_port_h = mock.sentinel.notif_port_h

        def fake_process_event(event):
            if event == mock.sentinel.ignored_event:
                return
            self._listener._running = False
            return mock.sentinel.processed_event
        mock_process_event.side_effect = fake_process_event
        self._listener._listen()
        processed_event = self._listener._event_queue.get(block=False)
        self.assertEqual(mock.sentinel.processed_event, processed_event)
        self.assertTrue(self._listener._event_queue.empty())
        self._clusapi.get_cluster_notify_v2.assert_any_call(mock.sentinel.notif_port_h, timeout_ms=-1)

    def test_listen_exception(self):
        self._clusapi.get_cluster_notify_v2.side_effect = test_base.TestingException
        self._listener._listen()
        self.assertFalse(self._listener._running)

    @mock.patch.object(clusterutils._ClusterEventListener, '_setup')
    @mock.patch.object(clusterutils.time, 'sleep')
    def test_listen_ignore_exception(self, mock_sleep, mock_setup):
        self._setup_listener(stop_on_error=False)
        self._clusapi.get_cluster_notify_v2.side_effect = (test_base.TestingException, KeyboardInterrupt)
        self.assertRaises(KeyboardInterrupt, self._listener._listen)
        self.assertTrue(self._listener._running)
        mock_sleep.assert_called_once_with(self._listener._error_sleep_interval)

    def test_get_event(self):
        self._listener._event_queue = mock.Mock()
        event = self._listener.get(timeout=mock.sentinel.timeout)
        self.assertEqual(self._listener._event_queue.get.return_value, event)
        self._listener._event_queue.get.assert_called_once_with(timeout=mock.sentinel.timeout)

    def test_get_event_listener_stopped(self):
        self._listener._running = False
        self.assertRaises(exceptions.OSWinException, self._listener.get, timeout=1)

        def fake_get(block=True, timeout=0):
            self._listener._running = False
            return None
        self._listener._running = True
        self._listener._event_queue = mock.Mock(get=fake_get)
        self.assertRaises(exceptions.OSWinException, self._listener.get, timeout=1)

    @mock.patch.object(clusterutils._ClusterEventListener, '_ensure_listener_running')
    @mock.patch.object(clusterutils._ClusterEventListener, 'stop')
    def test_context_manager(self, mock_stop, mock_ensure_running):
        with self._listener as li:
            self.assertIs(self._listener, li)
            mock_ensure_running.assert_called_once_with()
        mock_stop.assert_called_once_with()