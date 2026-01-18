import glob
import operator
import os
import shutil
import sys
import tempfile
from incremental import Version
from twisted.python import release
from twisted.python._release import (
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
class UtilityTests(ExternalTempdirTestCase):
    """
    Tests for various utility functions for releasing.
    """

    def test_chdir(self):
        """
        Test that the runChdirSafe is actually safe, i.e., it still
        changes back to the original directory even if an error is
        raised.
        """
        cwd = os.getcwd()

        def chAndBreak():
            os.mkdir('releaseCh')
            os.chdir('releaseCh')
            1 // 0
        self.assertRaises(ZeroDivisionError, release.runChdirSafe, chAndBreak)
        self.assertEqual(cwd, os.getcwd())

    def test_replaceInFile(self):
        """
        L{replaceInFile} replaces data in a file based on a dict. A key from
        the dict that is found in the file is replaced with the corresponding
        value.
        """
        content = 'foo\nhey hey $VER\nbar\n'
        with open('release.replace', 'w') as outf:
            outf.write(content)
        expected = content.replace('$VER', '2.0.0')
        replaceInFile('release.replace', {'$VER': '2.0.0'})
        with open('release.replace') as f:
            self.assertEqual(f.read(), expected)
        expected = expected.replace('2.0.0', '3.0.0')
        replaceInFile('release.replace', {'2.0.0': '3.0.0'})
        with open('release.replace') as f:
            self.assertEqual(f.read(), expected)