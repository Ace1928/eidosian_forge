import os
import shutil
import sys
import tempfile
import unittest
from os.path import join
from tempfile import TemporaryDirectory
from IPython.core.completerlib import magic_run_completer, module_completion, try_import
from IPython.testing.decorators import onlyif_unicode_paths
class Test_magic_run_completer_nonascii(unittest.TestCase):

    @onlyif_unicode_paths
    def setUp(self):
        self.BASETESTDIR = tempfile.mkdtemp()
        for fil in [u'aaø.py', u'a.py', u'b.py']:
            with open(join(self.BASETESTDIR, fil), 'w', encoding='utf-8') as sfile:
                sfile.write('pass\n')
        self.oldpath = os.getcwd()
        os.chdir(self.BASETESTDIR)

    def tearDown(self):
        os.chdir(self.oldpath)
        shutil.rmtree(self.BASETESTDIR)

    @onlyif_unicode_paths
    def test_1(self):
        """Test magic_run_completer, should match two alternatives
        """
        event = MockEvent(u'%run a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'a.py', u'aaø.py'})

    @onlyif_unicode_paths
    def test_2(self):
        """Test magic_run_completer, should match one alternative
        """
        event = MockEvent(u'%run aa')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'aaø.py'})

    @onlyif_unicode_paths
    def test_3(self):
        """Test magic_run_completer with unterminated " """
        event = MockEvent(u'%run "a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'a.py', u'aaø.py'})