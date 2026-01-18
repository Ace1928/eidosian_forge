import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe.NamedPipeHandler, '_rotate_logs')
def _test_write_to_log(self, mock_rotate_logs, size_exceeded=False):
    self._mock_setup_pipe_handler()
    self._handler._stopped.isSet.return_value = False
    fake_handle = self._handler._log_file_handle
    fake_handle.tell.return_value = constants.MAX_CONSOLE_LOG_FILE_SIZE if size_exceeded else 0
    fake_data = 'fake_data'
    self._handler._write_to_log(fake_data)
    if size_exceeded:
        mock_rotate_logs.assert_called_once_with()
    self._handler._log_file_handle.write.assert_called_once_with(fake_data)