import errno
import os
import shutil
import sys
from .. import tests, ui
from ..clean_tree import clean_tree, iter_deletables
from ..controldir import ControlDir
from ..osutils import supports_symlinks
from . import TestCaseInTempDir
class TestCleanTree(TestCaseInTempDir):

    def test_symlinks(self):
        if supports_symlinks(self.test_dir) is False:
            return
        os.mkdir('branch')
        ControlDir.create_standalone_workingtree('branch')
        os.symlink(os.path.realpath('no-die-please'), 'branch/die-please')
        os.mkdir('no-die-please')
        self.assertPathExists('branch/die-please')
        os.mkdir('no-die-please/child')
        clean_tree('branch', unknown=True, no_prompt=True)
        self.assertPathExists('no-die-please')
        self.assertPathExists('no-die-please/child')

    def test_iter_deletable(self):
        """Files are selected for deletion appropriately"""
        os.mkdir('branch')
        tree = ControlDir.create_standalone_workingtree('branch')
        transport = tree.controldir.root_transport
        transport.put_bytes('.bzrignore', b'*~\n*.pyc\n.bzrignore\n')
        transport.put_bytes('file.BASE', b'contents')
        with tree.lock_write():
            self.assertEqual(len(list(iter_deletables(tree, unknown=True))), 1)
            transport.put_bytes('file', b'contents')
            transport.put_bytes('file~', b'contents')
            transport.put_bytes('file.pyc', b'contents')
            dels = sorted([r for a, r in iter_deletables(tree, unknown=True)])
            self.assertEqual(['file', 'file.BASE'], dels)
            dels = [r for a, r in iter_deletables(tree, detritus=True)]
            self.assertEqual(sorted(['file~', 'file.BASE']), dels)
            dels = [r for a, r in iter_deletables(tree, ignored=True)]
            self.assertEqual(sorted(['file~', 'file.pyc', '.bzrignore']), dels)
            dels = [r for a, r in iter_deletables(tree, unknown=False)]
            self.assertEqual([], dels)

    def test_delete_items_warnings(self):
        """Ensure delete_items issues warnings on EACCES. (bug #430785)
        """

        def _dummy_unlink(path):
            """unlink() files other than files named '0foo'.
            """
            if path.endswith('0foo'):
                e = OSError()
                e.errno = errno.EACCES
                raise e

        def _dummy_rmtree(path, ignore_errors=False, onerror=None):
            """Call user supplied error handler onerror.
            """
            try:
                raise OSError
            except OSError as e:
                e.errno = errno.EACCES
                excinfo = sys.exc_info()
                function = os.remove
                if 'subdir0' not in path:
                    function = os.listdir
                onerror(function=function, path=path, excinfo=excinfo)
        self.overrideAttr(os, 'unlink', _dummy_unlink)
        self.overrideAttr(shutil, 'rmtree', _dummy_rmtree)
        ui.ui_factory = tests.TestUIFactory()
        stderr = ui.ui_factory.stderr
        ControlDir.create_standalone_workingtree('.')
        self.build_tree(['0foo', '1bar', '2baz', 'subdir0/'])
        clean_tree('.', unknown=True, no_prompt=True)
        self.assertContainsRe(stderr.getvalue(), 'bzr: warning: unable to remove.*0foo')
        self.assertContainsRe(stderr.getvalue(), 'bzr: warning: unable to remove.*subdir0')
        self.build_tree(['subdir1/'])
        self.assertRaises(OSError, clean_tree, '.', unknown=True, no_prompt=True)