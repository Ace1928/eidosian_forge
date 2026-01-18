from __future__ import annotations
import errno
import io
import os
import pickle
import stat
import sys
import time
from pprint import pformat
from typing import IO, AnyStr, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest import skipIf
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
from twisted.python import filepath
from twisted.python.filepath import FileMode, OtherAnyStr
from twisted.python.runtime import platform
from twisted.python.win32 import ERROR_DIRECTORY
from twisted.trial.unittest import SynchronousTestCase as TestCase
class AbstractFilePathTests(BytesTestCase):
    """
    Tests for L{IFilePath} implementations.
    """
    f1content = b'file 1'
    f2content = b'file 2'

    def _mkpath(self, *p: bytes) -> bytes:
        x = os.path.abspath(os.path.join(self.cmn, *p))
        self.all.append(x)
        return x

    def subdir(self, *dirname: bytes) -> None:
        os.mkdir(self._mkpath(*dirname))

    def subfile(self, *dirname: bytes) -> io.BufferedWriter:
        return open(self._mkpath(*dirname), 'wb')

    def setUp(self) -> None:
        self.now = time.time()
        cmn = self.cmn = os.path.abspath(self.mktemp())
        self.all = [cmn]
        os.mkdir(cmn)
        self.subdir(b'sub1')
        with self.subfile(b'file1') as f:
            f.write(self.f1content)
        with self.subfile(b'sub1', b'file2') as f:
            f.write(self.f2content)
        self.subdir(b'sub3')
        self.subfile(b'sub3', b'file3.ext1').close()
        self.subfile(b'sub3', b'file3.ext2').close()
        self.subfile(b'sub3', b'file3.ext3').close()
        self.path = filepath.FilePath(cmn)
        self.root = filepath.FilePath(b'/')

    def test_verifyObject(self) -> None:
        """
        Instances of the path type being tested provide L{IFilePath}.
        """
        self.assertTrue(verifyObject(filepath.IFilePath, self.path))

    def test_segmentsFromPositive(self) -> None:
        """
        Verify that the segments between two paths are correctly identified.
        """
        self.assertEqual(self.path.child(b'a').child(b'b').child(b'c').segmentsFrom(self.path), [b'a', b'b', b'c'])

    def test_segmentsFromNegative(self) -> None:
        """
        Verify that segmentsFrom notices when the ancestor isn't an ancestor.
        """
        self.assertRaises(ValueError, self.path.child(b'a').child(b'b').child(b'c').segmentsFrom, self.path.child(b'd').child(b'c').child(b'e'))

    def test_walk(self) -> None:
        """
        Verify that walking the path gives the same result as the known file
        hierarchy.
        """
        x = [foo.path for foo in self.path.walk()]
        self.assertEqual(set(x), set(self.all))

    def test_parents(self) -> None:
        """
        L{FilePath.parents()} should return an iterator of every ancestor of
        the L{FilePath} in question.
        """
        L = []
        pathobj = self.path.child(b'a').child(b'b').child(b'c')
        fullpath = pathobj.path
        lastpath = fullpath
        thispath = os.path.dirname(fullpath)
        while lastpath != self.root.path:
            L.append(thispath)
            lastpath = thispath
            thispath = os.path.dirname(thispath)
        self.assertEqual([x.path for x in pathobj.parents()], L)

    def test_validSubdir(self) -> None:
        """
        Verify that a valid subdirectory will show up as a directory, but not as a
        file, not as a symlink, and be listable.
        """
        sub1 = self.path.child(b'sub1')
        self.assertTrue(sub1.exists(), 'This directory does exist.')
        self.assertTrue(sub1.isdir(), "It's a directory.")
        self.assertFalse(sub1.isfile(), "It's a directory.")
        self.assertFalse(sub1.islink(), "It's a directory.")
        self.assertEqual(sub1.listdir(), [b'file2'])

    def test_invalidSubdir(self) -> None:
        """
        Verify that a subdirectory that doesn't exist is reported as such.
        """
        sub2 = self.path.child(b'sub2')
        self.assertFalse(sub2.exists(), 'This directory does not exist.')

    def test_validFiles(self) -> None:
        """
        Make sure that we can read existent non-empty files.
        """
        f1 = self.path.child(b'file1')
        with f1.open() as f:
            self.assertEqual(f.read(), self.f1content)
        f2 = self.path.child(b'sub1').child(b'file2')
        with f2.open() as f:
            self.assertEqual(f.read(), self.f2content)

    def test_multipleChildSegments(self) -> None:
        """
        C{fp.descendant([a, b, c])} returns the same L{FilePath} as is returned
        by C{fp.child(a).child(b).child(c)}.
        """
        multiple = self.path.descendant([b'a', b'b', b'c'])
        single = self.path.child(b'a').child(b'b').child(b'c')
        self.assertEqual(multiple, single)

    def test_dictionaryKeys(self) -> None:
        """
        Verify that path instances are usable as dictionary keys.
        """
        f1 = self.path.child(b'file1')
        f1prime = self.path.child(b'file1')
        f2 = self.path.child(b'file2')
        dictoid = {}
        dictoid[f1] = 3
        dictoid[f1prime] = 4
        self.assertEqual(dictoid[f1], 4)
        self.assertEqual(list(dictoid.keys()), [f1])
        self.assertIs(list(dictoid.keys())[0], f1)
        self.assertIsNot(list(dictoid.keys())[0], f1prime)
        dictoid[f2] = 5
        self.assertEqual(dictoid[f2], 5)
        self.assertEqual(len(dictoid), 2)

    def test_dictionaryKeyWithString(self) -> None:
        """
        Verify that path instances are usable as dictionary keys which do not clash
        with their string counterparts.
        """
        f1 = self.path.child(b'file1')
        dictoid: Dict[Union[filepath.FilePath[bytes], bytes], str] = {f1: 'hello'}
        dictoid[f1.path] = 'goodbye'
        self.assertEqual(len(dictoid), 2)

    def test_childrenNonexistentError(self) -> None:
        """
        Verify that children raises the appropriate exception for non-existent
        directories.
        """
        self.assertRaises(filepath.UnlistableError, self.path.child(b'not real').children)

    def test_childrenNotDirectoryError(self) -> None:
        """
        Verify that listdir raises the appropriate exception for attempting to list
        a file rather than a directory.
        """
        self.assertRaises(filepath.UnlistableError, self.path.child(b'file1').children)

    def test_newTimesAreFloats(self) -> None:
        """
        Verify that all times returned from the various new time functions are ints
        (and hopefully therefore 'high precision').
        """
        for p in (self.path, self.path.child(b'file1')):
            self.assertEqual(type(p.getAccessTime()), float)
            self.assertEqual(type(p.getModificationTime()), float)
            self.assertEqual(type(p.getStatusChangeTime()), float)

    def test_oldTimesAreInts(self) -> None:
        """
        Verify that all times returned from the various time functions are
        integers, for compatibility.
        """
        for p in (self.path, self.path.child(b'file1')):
            self.assertEqual(type(p.getatime()), int)
            self.assertEqual(type(p.getmtime()), int)
            self.assertEqual(type(p.getctime()), int)