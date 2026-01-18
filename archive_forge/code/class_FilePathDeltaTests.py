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
class FilePathDeltaTests(TestCase):
    """
    Tests for L{filePathDelta}.
    """

    def test_filePathDeltaSubdir(self):
        """
        L{filePathDelta} can create a simple relative path to a child path.
        """
        self.assertEqual(filePathDelta(FilePath('/foo/bar'), FilePath('/foo/bar/baz')), ['baz'])

    def test_filePathDeltaSiblingDir(self):
        """
        L{filePathDelta} can traverse upwards to create relative paths to
        siblings.
        """
        self.assertEqual(filePathDelta(FilePath('/foo/bar'), FilePath('/foo/baz')), ['..', 'baz'])

    def test_filePathNoCommonElements(self):
        """
        L{filePathDelta} can create relative paths to totally unrelated paths
        for maximum portability.
        """
        self.assertEqual(filePathDelta(FilePath('/foo/bar'), FilePath('/baz/quux')), ['..', '..', 'baz', 'quux'])

    def test_filePathDeltaSimilarEndElements(self):
        """
        L{filePathDelta} doesn't take into account final elements when
        comparing 2 paths, but stops at the first difference.
        """
        self.assertEqual(filePathDelta(FilePath('/foo/bar/bar/spam'), FilePath('/foo/bar/baz/spam')), ['..', '..', 'baz', 'spam'])