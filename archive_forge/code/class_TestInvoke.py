import os
import sys
import tempfile
from .. import mergetools, tests
class TestInvoke(tests.TestCaseInTempDir):

    def setUp(self):
        super(tests.TestCaseInTempDir, self).setUp()
        self._exe = None
        self._args = None
        self.build_tree_contents((('test.txt', b'stuff'), ('test.txt.BASE', b'base stuff'), ('test.txt.THIS', b'this stuff'), ('test.txt.OTHER', b'other stuff')))

    def test_invoke_expands_exe_path(self):
        self.overrideEnv('PATH', os.path.dirname(sys.executable))

        def dummy_invoker(exe, args, cleanup):
            self._exe = exe
            self._args = args
            cleanup(0)
            return 0
        command = '%s {result}' % os.path.basename(sys.executable)
        retcode = mergetools.invoke(command, 'test.txt', dummy_invoker)
        self.assertEqual(0, retcode)
        self.assertEqual(sys.executable, self._exe)
        self.assertEqual(['test.txt'], self._args)

    def test_success(self):

        def dummy_invoker(exe, args, cleanup):
            self._exe = exe
            self._args = args
            cleanup(0)
            return 0
        retcode = mergetools.invoke('tool {result}', 'test.txt', dummy_invoker)
        self.assertEqual(0, retcode)
        self.assertEqual('tool', self._exe)
        self.assertEqual(['test.txt'], self._args)

    def test_failure(self):

        def dummy_invoker(exe, args, cleanup):
            self._exe = exe
            self._args = args
            cleanup(1)
            return 1
        retcode = mergetools.invoke('tool {result}', 'test.txt', dummy_invoker)
        self.assertEqual(1, retcode)
        self.assertEqual('tool', self._exe)
        self.assertEqual(['test.txt'], self._args)

    def test_success_tempfile(self):

        def dummy_invoker(exe, args, cleanup):
            self._exe = exe
            self._args = args
            self.assertPathExists(args[0])
            with open(args[0], 'w') as f:
                f.write('temp stuff')
            cleanup(0)
            return 0
        retcode = mergetools.invoke('tool {this_temp}', 'test.txt', dummy_invoker)
        self.assertEqual(0, retcode)
        self.assertEqual('tool', self._exe)
        self.assertPathDoesNotExist(self._args[0])
        self.assertFileEqual(b'temp stuff', 'test.txt')

    def test_failure_tempfile(self):

        def dummy_invoker(exe, args, cleanup):
            self._exe = exe
            self._args = args
            self.assertPathExists(args[0])
            self.log(repr(args))
            with open(args[0], 'w') as f:
                self.log(repr(f))
                f.write('temp stuff')
            cleanup(1)
            return 1
        retcode = mergetools.invoke('tool {this_temp}', 'test.txt', dummy_invoker)
        self.assertEqual(1, retcode)
        self.assertEqual('tool', self._exe)
        self.assertFileEqual(b'stuff', 'test.txt')