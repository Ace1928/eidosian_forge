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
class ProcessExecutionErrorTest(test_base.BaseTestCase):

    def test_defaults(self):
        err = processutils.ProcessExecutionError()
        self.assertIn('None\n', str(err))
        self.assertIn('code: -\n', str(err))

    def test_with_description(self):
        description = 'The Narwhal Bacons at Midnight'
        err = processutils.ProcessExecutionError(description=description)
        self.assertIn(description, str(err))

    def test_with_exit_code(self):
        exit_code = 0
        err = processutils.ProcessExecutionError(exit_code=exit_code)
        self.assertIn(str(exit_code), str(err))

    def test_with_cmd(self):
        cmd = 'telinit'
        err = processutils.ProcessExecutionError(cmd=cmd)
        self.assertIn(cmd, str(err))

    def test_with_stdout(self):
        stdout = "\n        Lo, praise of the prowess of people-kings\n        of spear-armed Danes, in days long sped,\n        we have heard, and what honor the athelings won!\n        Oft Scyld the Scefing from squadroned foes,\n        from many a tribe, the mead-bench tore,\n        awing the earls. Since erst he lay\n        friendless, a foundling, fate repaid him:\n        for he waxed under welkin, in wealth he throve,\n        till before him the folk, both far and near,\n        who house by the whale-path, heard his mandate,\n        gave him gifts: a good king he!\n        To him an heir was afterward born,\n        a son in his halls, whom heaven sent\n        to favor the folk, feeling their woe\n        that erst they had lacked an earl for leader\n        so long a while; the Lord endowed him,\n        the Wielder of Wonder, with world's renown.\n        ".strip()
        err = processutils.ProcessExecutionError(stdout=stdout)
        print(str(err))
        self.assertIn('people-kings', str(err))

    def test_with_stderr(self):
        stderr = 'Cottonian library'
        err = processutils.ProcessExecutionError(stderr=stderr)
        self.assertIn(stderr, str(err))

    def test_retry_on_failure(self):
        fd, tmpfilename = tempfile.mkstemp()
        _, tmpfilename2 = tempfile.mkstemp()
        try:
            fp = os.fdopen(fd, 'w+')
            fp.write('#!/bin/sh\n# If stdin fails to get passed during one of the runs, make a note.\nif ! grep -q foo\nthen\n    echo \'failure\' > "$1"\nfi\n# If stdin has failed to get passed during this or a previous run, exit early.\nif grep failure "$1"\nthen\n    exit 1\nfi\nruns="$(cat $1)"\nif [ -z "$runs" ]\nthen\n    runs=0\nfi\nruns=$(($runs + 1))\necho $runs > "$1"\nexit 1\n')
            fp.close()
            os.chmod(tmpfilename, 493)
            self.assertRaises(processutils.ProcessExecutionError, processutils.execute, tmpfilename, tmpfilename2, attempts=10, process_input=b'foo', delay_on_retry=False)
            fp = open(tmpfilename2, 'r')
            runs = fp.read()
            fp.close()
            self.assertNotEqual('failure', 'stdin did not always get passed correctly', runs.strip())
            runs = int(runs.strip())
            self.assertEqual(10, runs, 'Ran %d times instead of 10.' % (runs,))
        finally:
            os.unlink(tmpfilename)
            os.unlink(tmpfilename2)

    def test_unknown_kwargs_raises_error(self):
        self.assertRaises(processutils.UnknownArgumentError, processutils.execute, '/usr/bin/env', 'true', this_is_not_a_valid_kwarg=True)

    def test_check_exit_code_boolean(self):
        processutils.execute('/usr/bin/env', 'false', check_exit_code=False)
        self.assertRaises(processutils.ProcessExecutionError, processutils.execute, '/usr/bin/env', 'false', check_exit_code=True)

    def test_check_cwd(self):
        tmpdir = tempfile.mkdtemp()
        out, err = processutils.execute('/usr/bin/env', 'sh', '-c', 'pwd', cwd=tmpdir)
        self.assertIn(tmpdir, out)

    def test_process_input_with_string(self):
        code = ';'.join(('import sys', 'print(len(sys.stdin.readlines()))'))
        args = [sys.executable, '-c', code]
        input = '\n'.join(['foo', 'bar', 'baz'])
        stdout, stderr = processutils.execute(*args, process_input=input)
        self.assertEqual('3', stdout.rstrip())

    def test_check_exit_code_list(self):
        processutils.execute('/usr/bin/env', 'sh', '-c', 'exit 101', check_exit_code=(101, 102))
        processutils.execute('/usr/bin/env', 'sh', '-c', 'exit 102', check_exit_code=(101, 102))
        self.assertRaises(processutils.ProcessExecutionError, processutils.execute, '/usr/bin/env', 'sh', '-c', 'exit 103', check_exit_code=(101, 102))
        self.assertRaises(processutils.ProcessExecutionError, processutils.execute, '/usr/bin/env', 'sh', '-c', 'exit 0', check_exit_code=(101, 102))

    def test_no_retry_on_success(self):
        fd, tmpfilename = tempfile.mkstemp()
        _, tmpfilename2 = tempfile.mkstemp()
        try:
            fp = os.fdopen(fd, 'w+')
            fp.write('#!/bin/sh\n# If we\'ve already run, bail out.\ngrep -q foo "$1" && exit 1\n# Mark that we\'ve run before.\necho foo > "$1"\n# Check that stdin gets passed correctly.\ngrep foo\n')
            fp.close()
            os.chmod(tmpfilename, 493)
            processutils.execute(tmpfilename, tmpfilename2, process_input=b'foo', attempts=2)
        finally:
            os.unlink(tmpfilename)
            os.unlink(tmpfilename2)

    def test_exception_on_communicate_error(self):
        mock = self.useFixture(fixtures.MockPatch('subprocess.Popen.communicate', side_effect=OSError(errno.EAGAIN, 'fake-test')))
        self.assertRaises(OSError, processutils.execute, '/usr/bin/env', 'false', check_exit_code=False)
        self.assertEqual(1, mock.mock.call_count)

    def test_retry_on_communicate_error(self):
        mock = self.useFixture(fixtures.MockPatch('subprocess.Popen.communicate', side_effect=OSError(errno.EAGAIN, 'fake-test')))
        self.assertRaises(OSError, processutils.execute, '/usr/bin/env', 'false', check_exit_code=False, attempts=5)
        self.assertEqual(5, mock.mock.call_count)

    def _test_and_check_logging_communicate_errors(self, log_errors=None, attempts=None):
        mock = self.useFixture(fixtures.MockPatch('subprocess.Popen.communicate', side_effect=OSError(errno.EAGAIN, 'fake-test')))
        fixture = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
        kwargs = {}
        if log_errors:
            kwargs.update({'log_errors': log_errors})
        if attempts:
            kwargs.update({'attempts': attempts})
        self.assertRaises(OSError, processutils.execute, '/usr/bin/env', 'false', **kwargs)
        self.assertEqual(attempts if attempts else 1, mock.mock.call_count)
        self.assertIn('Got an OSError', fixture.output)
        self.assertIn('errno: %d' % errno.EAGAIN, fixture.output)
        self.assertIn("'/usr/bin/env false'", fixture.output)

    def test_logging_on_communicate_error_1(self):
        self._test_and_check_logging_communicate_errors(log_errors=processutils.LOG_FINAL_ERROR, attempts=None)

    def test_logging_on_communicate_error_2(self):
        self._test_and_check_logging_communicate_errors(log_errors=processutils.LOG_FINAL_ERROR, attempts=1)

    def test_logging_on_communicate_error_3(self):
        self._test_and_check_logging_communicate_errors(log_errors=processutils.LOG_FINAL_ERROR, attempts=5)

    def test_logging_on_communicate_error_4(self):
        self._test_and_check_logging_communicate_errors(log_errors=processutils.LOG_ALL_ERRORS, attempts=None)

    def test_logging_on_communicate_error_5(self):
        self._test_and_check_logging_communicate_errors(log_errors=processutils.LOG_ALL_ERRORS, attempts=1)

    def test_logging_on_communicate_error_6(self):
        self._test_and_check_logging_communicate_errors(log_errors=processutils.LOG_ALL_ERRORS, attempts=5)

    def test_with_env_variables(self):
        env_vars = {'SUPER_UNIQUE_VAR': 'The answer is 42'}
        out, err = processutils.execute('/usr/bin/env', env_variables=env_vars)
        self.assertIsInstance(out, str)
        self.assertIsInstance(err, str)
        self.assertIn('SUPER_UNIQUE_VAR=The answer is 42', out)

    def test_binary(self):
        env_vars = {'SUPER_UNIQUE_VAR': 'The answer is 42'}
        out, err = processutils.execute('/usr/bin/env', env_variables=env_vars, binary=True)
        self.assertIsInstance(out, bytes)
        self.assertIsInstance(err, bytes)
        self.assertIn(b'SUPER_UNIQUE_VAR=The answer is 42', out)

    def test_exception_and_masking(self):
        tmpfilename = self.create_tempfiles([['test_exceptions_and_masking', TEST_EXCEPTION_AND_MASKING_SCRIPT]], ext='bash')[0]
        os.chmod(tmpfilename, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        err = self.assertRaises(processutils.ProcessExecutionError, processutils.execute, tmpfilename, 'password="secret"', 'something')
        self.assertEqual(38, err.exit_code)
        self.assertIsInstance(err.stdout, str)
        self.assertIsInstance(err.stderr, str)
        self.assertIn('onstdout --password="***"', err.stdout)
        self.assertIn('onstderr --password="***"', err.stderr)
        self.assertEqual(' '.join([tmpfilename, 'password="***"', 'something']), err.cmd)
        self.assertNotIn('secret', str(err))

    def execute_undecodable_bytes(self, out_bytes, err_bytes, exitcode=0, binary=False):
        code = ';'.join(('import sys', 'sys.stdout.buffer.write(%a)' % out_bytes, 'sys.stdout.flush()', 'sys.stderr.buffer.write(%a)' % err_bytes, 'sys.stderr.flush()', 'sys.exit(%s)' % exitcode))
        return processutils.execute(sys.executable, '-c', code, binary=binary)

    def check_undecodable_bytes(self, binary):
        out_bytes = b'out: ' + UNDECODABLE_BYTES
        err_bytes = b'err: ' + UNDECODABLE_BYTES
        out, err = self.execute_undecodable_bytes(out_bytes, err_bytes, binary=binary)
        if not binary:
            self.assertEqual(os.fsdecode(out_bytes), out)
            self.assertEqual(os.fsdecode(err_bytes), err)
        else:
            self.assertEqual(out, out_bytes)
            self.assertEqual(err, err_bytes)

    def test_undecodable_bytes(self):
        self.check_undecodable_bytes(False)

    def test_binary_undecodable_bytes(self):
        self.check_undecodable_bytes(True)

    def check_undecodable_bytes_error(self, binary):
        out_bytes = b'out: password="secret1" ' + UNDECODABLE_BYTES
        err_bytes = b'err: password="secret2" ' + UNDECODABLE_BYTES
        exc = self.assertRaises(processutils.ProcessExecutionError, self.execute_undecodable_bytes, out_bytes, err_bytes, exitcode=1, binary=binary)
        out = exc.stdout
        err = exc.stderr
        out_bytes = b'out: password="***" ' + UNDECODABLE_BYTES
        err_bytes = b'err: password="***" ' + UNDECODABLE_BYTES
        self.assertEqual(os.fsdecode(out_bytes), out)
        self.assertEqual(os.fsdecode(err_bytes), err)

    def test_undecodable_bytes_error(self):
        self.check_undecodable_bytes_error(False)

    def test_binary_undecodable_bytes_error(self):
        self.check_undecodable_bytes_error(True)

    def test_picklable(self):
        exc = processutils.ProcessExecutionError(stdout='my stdout', stderr='my stderr', exit_code=42, cmd='my cmd', description='my description')
        exc_message = str(exc)
        exc = pickle.loads(pickle.dumps(exc))
        self.assertEqual('my stdout', exc.stdout)
        self.assertEqual('my stderr', exc.stderr)
        self.assertEqual(42, exc.exit_code)
        self.assertEqual('my cmd', exc.cmd)
        self.assertEqual('my description', exc.description)
        self.assertEqual(str(exc), exc_message)

    def test_timeout(self):
        start = time.time()
        self.assertRaisesRegex(Exception, 'timed out after 1 seconds', processutils.execute, '/usr/bin/env', 'sh', '-c', 'sleep 10', timeout=1)
        self.assertLess(time.time(), start + 5)