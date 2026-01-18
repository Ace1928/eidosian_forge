import os
import re
import sys
import shutil
import warnings
import textwrap
import unittest
import tempfile
import subprocess
from distutils import ccompiler
import runtests
import Cython.Distutils.extension
import Cython.Distutils.old_build_ext as build_ext
from Cython.Debugger import Cygdb as cygdb
class GdbDebuggerTestCase(DebuggerTestCase):

    def setUp(self):
        if not test_gdb():
            return
        super(GdbDebuggerTestCase, self).setUp()
        prefix_code = textwrap.dedent('            python\n\n            import os\n            import sys\n            import traceback\n\n            def excepthook(type, value, tb):\n                traceback.print_exception(type, value, tb)\n                sys.stderr.flush()\n                sys.stdout.flush()\n                os._exit(1)\n\n            sys.excepthook = excepthook\n\n            # Have tracebacks end up on sys.stderr (gdb replaces sys.stderr\n            # with an object that calls gdb.write())\n            sys.stderr = sys.__stderr__\n\n            end\n            ')
        code = textwrap.dedent('            python\n\n            from Cython.Debugger.Tests import test_libcython_in_gdb\n            test_libcython_in_gdb.main(version=%r)\n\n            end\n            ' % (sys.version_info[:2],))
        self.gdb_command_file = cygdb.make_command_file(self.tempdir, prefix_code)
        with open(self.gdb_command_file, 'a') as f:
            f.write(code)
        args = ['gdb', '-batch', '-x', self.gdb_command_file, '-n', '--args', sys.executable, '-c', 'import codefile']
        paths = []
        path = os.environ.get('PYTHONPATH')
        if path:
            paths.append(path)
        paths.append(os.path.dirname(os.path.dirname(os.path.abspath(Cython.__file__))))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
        self.p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    def tearDown(self):
        if not test_gdb():
            return
        try:
            super(GdbDebuggerTestCase, self).tearDown()
            if self.p:
                try:
                    self.p.stdout.close()
                except:
                    pass
                try:
                    self.p.stderr.close()
                except:
                    pass
                self.p.wait()
        finally:
            os.remove(self.gdb_command_file)