import os
import shutil
import traceback
import unittest
import warnings
from contextlib import closing
import psutil
from psutil import BSD
from psutil import POSIX
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import super
from psutil._compat import u
from psutil.tests import APPVEYOR
from psutil.tests import ASCII_FS
from psutil.tests import CI_TESTING
from psutil.tests import HAS_CONNECTIONS_UNIX
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import INVALID_UNICODE_SUFFIX
from psutil.tests import PYPY
from psutil.tests import TESTFN_PREFIX
from psutil.tests import UNICODE_SUFFIX
from psutil.tests import PsutilTestCase
from psutil.tests import bind_unix_socket
from psutil.tests import chdir
from psutil.tests import copyload_shared_lib
from psutil.tests import create_py_exe
from psutil.tests import get_testfn
from psutil.tests import safe_mkdir
from psutil.tests import safe_rmpath
from psutil.tests import serialrun
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import terminate
@serialrun
@unittest.skipIf(ASCII_FS, 'ASCII fs')
@unittest.skipIf(PYPY and (not PY3), 'too much trouble on PYPY2')
class TestFSAPIs(BaseUnicodeTest):
    """Test FS APIs with a funky, valid, UTF8 path name."""
    funky_suffix = UNICODE_SUFFIX

    def expect_exact_path_match(self):
        here = '.' if isinstance(self.funky_name, str) else u('.')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.funky_name in os.listdir(here)

    def test_proc_exe(self):
        cmd = [self.funky_name, '-c', 'import time; time.sleep(10)']
        subp = self.spawn_testproc(cmd)
        p = psutil.Process(subp.pid)
        exe = p.exe()
        self.assertIsInstance(exe, str)
        if self.expect_exact_path_match():
            self.assertEqual(os.path.normcase(exe), os.path.normcase(self.funky_name))

    def test_proc_name(self):
        cmd = [self.funky_name, '-c', 'import time; time.sleep(10)']
        subp = self.spawn_testproc(cmd)
        name = psutil.Process(subp.pid).name()
        self.assertIsInstance(name, str)
        if self.expect_exact_path_match():
            self.assertEqual(name, os.path.basename(self.funky_name))

    def test_proc_cmdline(self):
        cmd = [self.funky_name, '-c', 'import time; time.sleep(10)']
        subp = self.spawn_testproc(cmd)
        p = psutil.Process(subp.pid)
        cmdline = p.cmdline()
        for part in cmdline:
            self.assertIsInstance(part, str)
        if self.expect_exact_path_match():
            self.assertEqual(cmdline, cmd)

    def test_proc_cwd(self):
        dname = self.funky_name + '2'
        self.addCleanup(safe_rmpath, dname)
        safe_mkdir(dname)
        with chdir(dname):
            p = psutil.Process()
            cwd = p.cwd()
        self.assertIsInstance(p.cwd(), str)
        if self.expect_exact_path_match():
            self.assertEqual(cwd, dname)

    @unittest.skipIf(PYPY and WINDOWS, 'fails on PYPY + WINDOWS')
    def test_proc_open_files(self):
        p = psutil.Process()
        start = set(p.open_files())
        with open(self.funky_name, 'rb'):
            new = set(p.open_files())
        path = (new - start).pop().path
        self.assertIsInstance(path, str)
        if BSD and (not path):
            return self.skipTest('open_files on BSD is broken')
        if self.expect_exact_path_match():
            self.assertEqual(os.path.normcase(path), os.path.normcase(self.funky_name))

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_proc_connections(self):
        name = self.get_testfn(suffix=self.funky_suffix)
        try:
            sock = bind_unix_socket(name)
        except UnicodeEncodeError:
            if PY3:
                raise
            else:
                raise unittest.SkipTest('not supported')
        with closing(sock):
            conn = psutil.Process().connections('unix')[0]
            self.assertIsInstance(conn.laddr, str)
            self.assertEqual(conn.laddr, name)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @unittest.skipIf(not HAS_CONNECTIONS_UNIX, "can't list UNIX sockets")
    @skip_on_access_denied()
    def test_net_connections(self):

        def find_sock(cons):
            for conn in cons:
                if os.path.basename(conn.laddr).startswith(TESTFN_PREFIX):
                    return conn
            raise ValueError('connection not found')
        name = self.get_testfn(suffix=self.funky_suffix)
        try:
            sock = bind_unix_socket(name)
        except UnicodeEncodeError:
            if PY3:
                raise
            else:
                raise unittest.SkipTest('not supported')
        with closing(sock):
            cons = psutil.net_connections(kind='unix')
            conn = find_sock(cons)
            self.assertIsInstance(conn.laddr, str)
            self.assertEqual(conn.laddr, name)

    def test_disk_usage(self):
        dname = self.funky_name + '2'
        self.addCleanup(safe_rmpath, dname)
        safe_mkdir(dname)
        psutil.disk_usage(dname)

    @unittest.skipIf(not HAS_MEMORY_MAPS, 'not supported')
    @unittest.skipIf(not PY3, 'ctypes does not support unicode on PY2')
    @unittest.skipIf(PYPY, 'unstable on PYPY')
    def test_memory_maps(self):
        with copyload_shared_lib(suffix=self.funky_suffix) as funky_path:

            def normpath(p):
                return os.path.realpath(os.path.normcase(p))
            libpaths = [normpath(x.path) for x in psutil.Process().memory_maps()]
            libpaths = [x for x in libpaths if TESTFN_PREFIX in x]
            self.assertIn(normpath(funky_path), libpaths)
            for path in libpaths:
                self.assertIsInstance(path, str)