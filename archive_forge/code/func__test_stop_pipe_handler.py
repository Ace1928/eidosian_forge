import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe.NamedPipeHandler, '_cleanup_handles')
@mock.patch.object(namedpipe.NamedPipeHandler, '_cancel_io')
def _test_stop_pipe_handler(self, mock_cancel_io, mock_cleanup_handles, workers_started=True):
    self._mock_setup_pipe_handler()
    if not workers_started:
        handler_workers = []
        self._handler._workers = handler_workers
    else:
        handler_workers = self._handler._workers
        self._r_worker.is_alive.side_effect = (True, False)
        self._w_worker.is_alive.return_value = False
    self._handler.stop()
    self._handler._stopped.set.assert_called_once_with()
    if not workers_started:
        mock_cleanup_handles.assert_called_once_with()
    else:
        self.assertFalse(mock_cleanup_handles.called)
    if workers_started:
        mock_cancel_io.assert_called_once_with()
        self._r_worker.join.assert_called_once_with(0.5)
        self.assertFalse(self._w_worker.join.called)
    self.assertEqual([], self._handler._workers)