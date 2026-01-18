import threading
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from os_brick import executor as brick_executor
from os_brick.privileged import rootwrap
from os_brick.tests import base
class TestExecutor(base.TestCase):

    def test_default_execute(self):
        executor = brick_executor.Executor(root_helper=None)
        self.assertEqual(rootwrap.execute, executor._Executor__execute)

    def test_none_execute(self):
        executor = brick_executor.Executor(root_helper=None, execute=None)
        self.assertEqual(rootwrap.execute, executor._Executor__execute)

    def test_fake_execute(self):
        mock_execute = mock.Mock()
        executor = brick_executor.Executor(root_helper=None, execute=mock_execute)
        self.assertEqual(mock_execute, executor._Executor__execute)

    @mock.patch('sys.stdin', encoding='UTF-8')
    @mock.patch('os_brick.executor.priv_rootwrap.execute')
    def test_execute_non_safe_str_exception(self, execute_mock, stdin_mock):
        execute_mock.side_effect = putils.ProcessExecutionError(stdout='España', stderr='Zürich')
        executor = brick_executor.Executor(root_helper=None)
        exc = self.assertRaises(putils.ProcessExecutionError, executor._execute)
        self.assertEqual('España', exc.stdout)
        self.assertEqual('Zürich', exc.stderr)

    @mock.patch('sys.stdin', encoding='UTF-8')
    @mock.patch('os_brick.executor.priv_rootwrap.execute')
    def test_execute_non_safe_str(self, execute_mock, stdin_mock):
        execute_mock.return_value = ('España', 'Zürich')
        executor = brick_executor.Executor(root_helper=None)
        stdout, stderr = executor._execute()
        self.assertEqual('España', stdout)
        self.assertEqual('Zürich', stderr)

    @mock.patch('sys.stdin', encoding='UTF-8')
    @mock.patch('os_brick.executor.priv_rootwrap.execute')
    def test_execute_non_safe_bytes_exception(self, execute_mock, stdin_mock):
        execute_mock.side_effect = putils.ProcessExecutionError(stdout=bytes('España', 'utf-8'), stderr=bytes('Zürich', 'utf-8'))
        executor = brick_executor.Executor(root_helper=None)
        exc = self.assertRaises(putils.ProcessExecutionError, executor._execute)
        self.assertEqual('España', exc.stdout)
        self.assertEqual('Zürich', exc.stderr)

    @mock.patch('sys.stdin', encoding='UTF-8')
    @mock.patch('os_brick.executor.priv_rootwrap.execute')
    def test_execute_non_safe_bytes(self, execute_mock, stdin_mock):
        execute_mock.return_value = (bytes('España', 'utf-8'), bytes('Zürich', 'utf-8'))
        executor = brick_executor.Executor(root_helper=None)
        stdout, stderr = executor._execute()
        self.assertEqual('España', stdout)
        self.assertEqual('Zürich', stderr)