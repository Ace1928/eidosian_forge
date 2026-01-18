import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestCopyTree(tests.TestCaseInTempDir):

    def test_copy_basic_tree(self):
        self.build_tree(['source/', 'source/a', 'source/b/', 'source/b/c'])
        osutils.copy_tree('source', 'target')
        self.assertEqual(['a', 'b'], sorted(os.listdir('target')))
        self.assertEqual(['c'], os.listdir('target/b'))

    def test_copy_tree_target_exists(self):
        self.build_tree(['source/', 'source/a', 'source/b/', 'source/b/c', 'target/'])
        osutils.copy_tree('source', 'target')
        self.assertEqual(['a', 'b'], sorted(os.listdir('target')))
        self.assertEqual(['c'], os.listdir('target/b'))

    def test_copy_tree_symlinks(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        self.build_tree(['source/'])
        os.symlink('a/generic/path', 'source/lnk')
        osutils.copy_tree('source', 'target')
        self.assertEqual(['lnk'], os.listdir('target'))
        self.assertEqual('a/generic/path', os.readlink('target/lnk'))

    def test_copy_tree_handlers(self):
        processed_files = []
        processed_links = []

        def file_handler(from_path, to_path):
            processed_files.append(('f', from_path, to_path))

        def dir_handler(from_path, to_path):
            processed_files.append(('d', from_path, to_path))

        def link_handler(from_path, to_path):
            processed_links.append((from_path, to_path))
        handlers = {'file': file_handler, 'directory': dir_handler, 'symlink': link_handler}
        self.build_tree(['source/', 'source/a', 'source/b/', 'source/b/c'])
        if osutils.supports_symlinks(self.test_dir):
            os.symlink('a/generic/path', 'source/lnk')
        osutils.copy_tree('source', 'target', handlers=handlers)
        self.assertEqual([('d', 'source', 'target'), ('f', 'source/a', 'target/a'), ('d', 'source/b', 'target/b'), ('f', 'source/b/c', 'target/b/c')], processed_files)
        self.assertPathDoesNotExist('target')
        if osutils.supports_symlinks(self.test_dir):
            self.assertEqual([('source/lnk', 'target/lnk')], processed_links)