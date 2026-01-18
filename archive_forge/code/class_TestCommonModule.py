import ast
import collections
import errno
import json
import os
import pickle
import socket
import stat
import unittest
import psutil
import psutil.tests
from psutil import LINUX
from psutil import POSIX
from psutil import WINDOWS
from psutil._common import bcat
from psutil._common import cat
from psutil._common import debug
from psutil._common import isfile_strict
from psutil._common import memoize
from psutil._common import memoize_when_activated
from psutil._common import parse_environ_block
from psutil._common import supports_ipv6
from psutil._common import wrap_numbers
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import redirect_stderr
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import SCRIPTS_DIR
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import sh
class TestCommonModule(PsutilTestCase):

    def test_memoize_when_activated(self):

        class Foo:

            @memoize_when_activated
            def foo(self):
                calls.append(None)
        f = Foo()
        calls = []
        f.foo()
        f.foo()
        self.assertEqual(len(calls), 2)
        calls = []
        f.foo.cache_activate(f)
        f.foo()
        f.foo()
        self.assertEqual(len(calls), 1)
        calls = []
        f.foo.cache_deactivate(f)
        f.foo()
        f.foo()
        self.assertEqual(len(calls), 2)

    def test_parse_environ_block(self):

        def k(s):
            return s.upper() if WINDOWS else s
        self.assertEqual(parse_environ_block('a=1\x00'), {k('a'): '1'})
        self.assertEqual(parse_environ_block('a=1\x00b=2\x00\x00'), {k('a'): '1', k('b'): '2'})
        self.assertEqual(parse_environ_block('a=1\x00b=\x00\x00'), {k('a'): '1', k('b'): ''})
        self.assertEqual(parse_environ_block('a=1\x00b=2\x00\x00c=3\x00'), {k('a'): '1', k('b'): '2'})
        self.assertEqual(parse_environ_block('xxx\x00a=1\x00'), {k('a'): '1'})
        self.assertEqual(parse_environ_block('a=1\x00=b=2\x00'), {k('a'): '1'})
        self.assertEqual(parse_environ_block('a=1\x00b=2'), {k('a'): '1'})

    def test_supports_ipv6(self):
        self.addCleanup(supports_ipv6.cache_clear)
        if supports_ipv6():
            with mock.patch('psutil._common.socket') as s:
                s.has_ipv6 = False
                supports_ipv6.cache_clear()
                assert not supports_ipv6()
            supports_ipv6.cache_clear()
            with mock.patch('psutil._common.socket.socket', side_effect=socket.error) as s:
                assert not supports_ipv6()
                assert s.called
            supports_ipv6.cache_clear()
            with mock.patch('psutil._common.socket.socket', side_effect=socket.gaierror) as s:
                assert not supports_ipv6()
                supports_ipv6.cache_clear()
                assert s.called
            supports_ipv6.cache_clear()
            with mock.patch('psutil._common.socket.socket.bind', side_effect=socket.gaierror) as s:
                assert not supports_ipv6()
                supports_ipv6.cache_clear()
                assert s.called
        else:
            with self.assertRaises(socket.error):
                sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
                try:
                    sock.bind(('::1', 0))
                finally:
                    sock.close()

    def test_isfile_strict(self):
        this_file = os.path.abspath(__file__)
        assert isfile_strict(this_file)
        assert not isfile_strict(os.path.dirname(this_file))
        with mock.patch('psutil._common.os.stat', side_effect=OSError(errno.EPERM, 'foo')):
            self.assertRaises(OSError, isfile_strict, this_file)
        with mock.patch('psutil._common.os.stat', side_effect=OSError(errno.EACCES, 'foo')):
            self.assertRaises(OSError, isfile_strict, this_file)
        with mock.patch('psutil._common.os.stat', side_effect=OSError(errno.ENOENT, 'foo')):
            assert not isfile_strict(this_file)
        with mock.patch('psutil._common.stat.S_ISREG', return_value=False):
            assert not isfile_strict(this_file)

    def test_debug(self):
        if PY3:
            from io import StringIO
        else:
            from StringIO import StringIO
        with redirect_stderr(StringIO()) as f:
            debug('hello')
        msg = f.getvalue()
        assert msg.startswith('psutil-debug'), msg
        self.assertIn('hello', msg)
        self.assertIn(__file__.replace('.pyc', '.py'), msg)
        with redirect_stderr(StringIO()) as f:
            debug(ValueError('this is an error'))
        msg = f.getvalue()
        self.assertIn('ignoring ValueError', msg)
        self.assertIn("'this is an error'", msg)
        with redirect_stderr(StringIO()) as f:
            exc = OSError(2, 'no such file')
            exc.filename = '/foo'
            debug(exc)
        msg = f.getvalue()
        self.assertIn('no such file', msg)
        self.assertIn('/foo', msg)

    def test_cat_bcat(self):
        testfn = self.get_testfn()
        with open(testfn, 'w') as f:
            f.write('foo')
        self.assertEqual(cat(testfn), 'foo')
        self.assertEqual(bcat(testfn), b'foo')
        self.assertRaises(FileNotFoundError, cat, testfn + '-invalid')
        self.assertRaises(FileNotFoundError, bcat, testfn + '-invalid')
        self.assertEqual(cat(testfn + '-invalid', fallback='bar'), 'bar')
        self.assertEqual(bcat(testfn + '-invalid', fallback='bar'), 'bar')