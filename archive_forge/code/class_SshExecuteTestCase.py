import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
class SshExecuteTestCase(test_base.BaseTestCase):

    def test_invalid_addl_env(self):
        self.assertRaises(processutils.InvalidArgumentError, processutils.ssh_execute, None, 'ls', addl_env='important')

    def test_invalid_process_input(self):
        self.assertRaises(processutils.InvalidArgumentError, processutils.ssh_execute, None, 'ls', process_input='important')

    def test_timeout_error(self):
        self.assertRaises(socket.timeout, processutils.ssh_execute, FakeSshConnection(0), 'ls', timeout=10)

    def test_works(self):
        out, err = processutils.ssh_execute(FakeSshConnection(0), 'ls')
        self.assertEqual('stdout', out)
        self.assertEqual('stderr', err)
        self.assertIsInstance(out, str)
        self.assertIsInstance(err, str)

    def test_binary(self):
        o, e = processutils.ssh_execute(FakeSshConnection(0), 'ls', binary=True)
        self.assertEqual(b'stdout', o)
        self.assertEqual(b'stderr', e)
        self.assertIsInstance(o, bytes)
        self.assertIsInstance(e, bytes)

    def check_undecodable_bytes(self, binary):
        out_bytes = b'out: ' + UNDECODABLE_BYTES
        err_bytes = b'err: ' + UNDECODABLE_BYTES
        conn = FakeSshConnection(0, out=out_bytes, err=err_bytes)
        out, err = processutils.ssh_execute(conn, 'ls', binary=binary)
        if not binary:
            self.assertEqual(os.fsdecode(out_bytes), out)
            self.assertEqual(os.fsdecode(err_bytes), err)
        else:
            self.assertEqual(out_bytes, out)
            self.assertEqual(err_bytes, err)

    def test_undecodable_bytes(self):
        self.check_undecodable_bytes(False)

    def test_binary_undecodable_bytes(self):
        self.check_undecodable_bytes(True)

    def check_undecodable_bytes_error(self, binary):
        out_bytes = b'out: password="secret1" ' + UNDECODABLE_BYTES
        err_bytes = b'err: password="secret2" ' + UNDECODABLE_BYTES
        conn = FakeSshConnection(1, out=out_bytes, err=err_bytes)
        out_bytes = b'out: password="***" ' + UNDECODABLE_BYTES
        err_bytes = b'err: password="***" ' + UNDECODABLE_BYTES
        exc = self.assertRaises(processutils.ProcessExecutionError, processutils.ssh_execute, conn, 'ls', binary=binary, check_exit_code=True)
        out = exc.stdout
        err = exc.stderr
        self.assertEqual(os.fsdecode(out_bytes), out)
        self.assertEqual(os.fsdecode(err_bytes), err)

    def test_undecodable_bytes_error(self):
        self.check_undecodable_bytes_error(False)

    def test_binary_undecodable_bytes_error(self):
        self.check_undecodable_bytes_error(True)

    def test_fails(self):
        self.assertRaises(processutils.ProcessExecutionError, processutils.ssh_execute, FakeSshConnection(1), 'ls')

    def _test_compromising_ssh(self, rc, check):
        fixture = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
        fake_stdin = io.BytesIO()
        fake_stdout = mock.Mock()
        fake_stdout.channel.recv_exit_status.return_value = rc
        fake_stdout.read.return_value = b'password="secret"'
        fake_stderr = mock.Mock()
        fake_stderr.read.return_value = b'password="foobar"'
        command = 'ls --password="bar"'
        connection = mock.Mock()
        connection.exec_command.return_value = (fake_stdin, fake_stdout, fake_stderr)
        if check and rc != -1 and (rc != 0):
            err = self.assertRaises(processutils.ProcessExecutionError, processutils.ssh_execute, connection, command, check_exit_code=check)
            self.assertEqual(rc, err.exit_code)
            self.assertEqual('password="***"', err.stdout)
            self.assertEqual('password="***"', err.stderr)
            self.assertEqual('ls --password="***"', err.cmd)
            self.assertNotIn('secret', str(err))
            self.assertNotIn('foobar', str(err))
            err = self.assertRaises(processutils.ProcessExecutionError, processutils.ssh_execute, connection, command, check_exit_code=check, sanitize_stdout=False)
            self.assertEqual(rc, err.exit_code)
            self.assertEqual('password="***"', err.stdout)
            self.assertEqual('password="***"', err.stderr)
            self.assertEqual('ls --password="***"', err.cmd)
            self.assertNotIn('secret', str(err))
            self.assertNotIn('foobar', str(err))
        else:
            o, e = processutils.ssh_execute(connection, command, check_exit_code=check)
            self.assertEqual('password="***"', o)
            self.assertEqual('password="***"', e)
            self.assertIn('password="***"', fixture.output)
            self.assertNotIn('bar', fixture.output)
            o, e = processutils.ssh_execute(connection, command, check_exit_code=check, sanitize_stdout=False)
            self.assertEqual('password="secret"', o)
            self.assertEqual('password="***"', e)
            self.assertIn('password="***"', fixture.output)
            self.assertNotIn('bar', fixture.output)

    def test_compromising_ssh1(self):
        self._test_compromising_ssh(rc=-1, check=True)

    def test_compromising_ssh2(self):
        self._test_compromising_ssh(rc=0, check=True)

    def test_compromising_ssh3(self):
        self._test_compromising_ssh(rc=1, check=True)

    def test_compromising_ssh4(self):
        self._test_compromising_ssh(rc=1, check=False)

    def test_compromising_ssh5(self):
        self._test_compromising_ssh(rc=0, check=False)

    def test_compromising_ssh6(self):
        self._test_compromising_ssh(rc=-1, check=False)