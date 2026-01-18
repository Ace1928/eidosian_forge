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
class FilePathTests(AbstractFilePathTests):
    """
    Test various L{FilePath} path manipulations.

    In particular, note that tests defined on this class instead of on the base
    class are only run against L{twisted.python.filepath}.
    """

    def test_chmod(self) -> None:
        """
        L{FilePath.chmod} modifies the permissions of
        the passed file as expected (using C{os.stat} to check). We use some
        basic modes that should work everywhere (even on Windows).
        """
        for mode in (365, 511):
            self.path.child(b'sub1').chmod(mode)
            self.assertEqual(stat.S_IMODE(os.stat(self.path.child(b'sub1').path).st_mode), mode)

    def createLinks(self) -> None:
        """
        Create several symbolic links to files and directories.
        """
        subdir = self.path.child(b'sub1')
        os.symlink(subdir.path, self._mkpath(b'sub1.link'))
        os.symlink(subdir.child(b'file2').path, self._mkpath(b'file2.link'))
        os.symlink(subdir.child(b'file2').path, self._mkpath(b'sub1', b'sub1.file2.link'))

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_realpathSymlink(self) -> None:
        """
        L{FilePath.realpath} returns the path of the ultimate target of a
        symlink.
        """
        self.createLinks()
        os.symlink(self.path.child(b'file2.link').path, self.path.child(b'link.link').path)
        self.assertEqual(self.path.child(b'link.link').realpath(), self.path.child(b'sub1').child(b'file2'))

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_realpathCyclicalSymlink(self) -> None:
        """
        L{FilePath.realpath} raises L{filepath.LinkError} if the path is a
        symbolic link which is part of a cycle.
        """
        os.symlink(self.path.child(b'link1').path, self.path.child(b'link2').path)
        os.symlink(self.path.child(b'link2').path, self.path.child(b'link1').path)
        self.assertRaises(filepath.LinkError, self.path.child(b'link2').realpath)

    def test_realpathNoSymlink(self) -> None:
        """
        L{FilePath.realpath} returns the path itself if the path is not a
        symbolic link.
        """
        self.assertEqual(self.path.child(b'sub1').realpath(), self.path.child(b'sub1'))

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_walkCyclicalSymlink(self) -> None:
        """
        Verify that walking a path with a cyclical symlink raises an error
        """
        self.createLinks()
        os.symlink(self.path.child(b'sub1').path, self.path.child(b'sub1').child(b'sub1.loopylink').path)

        def iterateOverPath() -> List[bytes]:
            return [foo.path for foo in self.path.walk()]
        self.assertRaises(filepath.LinkError, iterateOverPath)

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_walkObeysDescendWithCyclicalSymlinks(self) -> None:
        """
        Verify that, after making a path with cyclical symlinks, when the
        supplied C{descend} predicate returns C{False}, the target is not
        traversed, as if it was a simple symlink.
        """
        self.createLinks()
        os.symlink(self.path.child(b'sub1').path, self.path.child(b'sub1').child(b'sub1.loopylink').path)

        def noSymLinks(path: filepath.FilePath[bytes]) -> bool:
            return not path.islink()

        def iterateOverPath() -> List[bytes]:
            return [foo.path for foo in self.path.walk(descend=noSymLinks)]
        self.assertTrue(iterateOverPath())

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_walkObeysDescend(self) -> None:
        """
        Verify that when the supplied C{descend} predicate returns C{False},
        the target is not traversed.
        """
        self.createLinks()

        def noSymLinks(path: filepath.FilePath[bytes]) -> bool:
            return not path.islink()
        x = [foo.path for foo in self.path.walk(descend=noSymLinks)]
        self.assertEqual(set(x), set(self.all))

    def test_getAndSet(self) -> None:
        content = b'newcontent'
        self.path.child(b'new').setContent(content)
        newcontent = self.path.child(b'new').getContent()
        self.assertEqual(content, newcontent)
        content = b'content'
        self.path.child(b'new').setContent(content, b'.tmp')
        newcontent = self.path.child(b'new').getContent()
        self.assertEqual(content, newcontent)

    def test_getContentFileClosing(self) -> None:
        """
        If reading from the underlying file raises an exception,
        L{FilePath.getContent} raises that exception after closing the file.
        """
        fp = ExplodingFilePath('')
        self.assertRaises(IOError, fp.getContent)
        self.assertTrue(fp.fp.closed)

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_symbolicLink(self) -> None:
        """
        Verify the behavior of the C{isLink} method against links and
        non-links. Also check that the symbolic link shares the directory
        property with its target.
        """
        s4 = self.path.child(b'sub4')
        s3 = self.path.child(b'sub3')
        os.symlink(s3.path, s4.path)
        self.assertTrue(s4.islink())
        self.assertFalse(s3.islink())
        self.assertTrue(s4.isdir())
        self.assertTrue(s3.isdir())

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_linkTo(self) -> None:
        """
        Verify that symlink creates a valid symlink that is both a link and a
        file if its target is a file, or a directory if its target is a
        directory.
        """
        targetLinks = [(self.path.child(b'sub2'), self.path.child(b'sub2.link')), (self.path.child(b'sub2').child(b'file3.ext1'), self.path.child(b'file3.ext1.link'))]
        for target, link in targetLinks:
            target.linkTo(link)
            self.assertTrue(link.islink(), 'This is a link')
            self.assertEqual(target.isdir(), link.isdir())
            self.assertEqual(target.isfile(), link.isfile())

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_linkToErrors(self) -> None:
        """
        Verify C{linkTo} fails in the following case:
            - the target is in a directory that doesn't exist
            - the target already exists
        """
        self.assertRaises(OSError, self.path.child(b'file1').linkTo, self.path.child(b'nosub').child(b'file1'))
        self.assertRaises(OSError, self.path.child(b'file1').linkTo, self.path.child(b'sub1').child(b'file2'))

    def testMultiExt(self) -> None:
        f3 = self.path.child(b'sub3').child(b'file3')
        exts = (b'.foo', b'.bar', b'ext1', b'ext2', b'ext3')
        self.assertFalse(f3.siblingExtensionSearch(*exts))
        f3e = f3.siblingExtension(b'.foo')
        f3e.touch()
        found = f3.siblingExtensionSearch(*exts)
        assert found is not None
        self.assertFalse(not found.exists())
        globbed = f3.siblingExtensionSearch(b'*')
        assert globbed is not None
        self.assertFalse(not globbed.exists())
        f3e.remove()
        self.assertFalse(f3.siblingExtensionSearch(*exts))

    def testPreauthChild(self) -> None:
        fp = filepath.FilePath(b'.')
        fp.preauthChild(b'foo/bar')
        self.assertRaises(filepath.InsecurePath, fp.preauthChild, '/mon€y')

    def testStatCache(self) -> None:
        p = self.path.child(b'stattest')
        p.touch()
        self.assertEqual(p.getsize(), 0)
        self.assertEqual(abs(p.getmtime() - time.time()) // 20, 0)
        self.assertEqual(abs(p.getctime() - time.time()) // 20, 0)
        self.assertEqual(abs(p.getatime() - time.time()) // 20, 0)
        self.assertTrue(p.exists())
        self.assertTrue(p.exists())
        os.remove(p.path)
        self.assertTrue(p.exists())
        p.restat(reraise=False)
        self.assertFalse(p.exists())
        self.assertFalse(p.islink())
        self.assertFalse(p.isdir())
        self.assertFalse(p.isfile())

    def testPersist(self) -> None:
        newpath = pickle.loads(pickle.dumps(self.path))
        self.assertEqual(self.path.__class__, newpath.__class__)
        self.assertEqual(self.path.path, newpath.path)

    def testInsecureUNIX(self) -> None:
        self.assertRaises(filepath.InsecurePath, self.path.child, b'..')
        self.assertRaises(filepath.InsecurePath, self.path.child, b'/etc')
        self.assertRaises(filepath.InsecurePath, self.path.child, b'../..')

    @skipIf(platform.getType() != 'win32', 'Test will run only on Windows.')
    def testInsecureWin32(self) -> None:
        self.assertRaises(filepath.InsecurePath, self.path.child, b'..\\..')
        self.assertRaises(filepath.InsecurePath, self.path.child, b'C:randomfile')

    @skipIf(platform.getType() != 'win32', 'Test will run only on Windows.')
    def testInsecureWin32Whacky(self) -> None:
        """
        Windows has 'special' filenames like NUL and CON and COM1 and LPR
        and PRN and ... god knows what else.  They can be located anywhere in
        the filesystem.  For obvious reasons, we do not wish to normally permit
        access to these.
        """
        self.assertRaises(filepath.InsecurePath, self.path.child, b'CON')
        self.assertRaises(filepath.InsecurePath, self.path.child, b'C:CON')
        self.assertRaises(filepath.InsecurePath, self.path.child, 'C:\\CON')

    def testComparison(self) -> None:
        self.assertEqual(filepath.FilePath(b'a'), filepath.FilePath(b'a'))
        self.assertTrue(filepath.FilePath(b'z') > filepath.FilePath(b'a'))
        self.assertTrue(filepath.FilePath(b'z') >= filepath.FilePath(b'a'))
        self.assertTrue(filepath.FilePath(b'a') >= filepath.FilePath(b'a'))
        self.assertTrue(filepath.FilePath(b'a') <= filepath.FilePath(b'a'))
        self.assertTrue(filepath.FilePath(b'a') < filepath.FilePath(b'z'))
        self.assertTrue(filepath.FilePath(b'a') <= filepath.FilePath(b'z'))
        self.assertTrue(filepath.FilePath(b'a') != filepath.FilePath(b'z'))
        self.assertTrue(filepath.FilePath(b'z') != filepath.FilePath(b'a'))
        self.assertFalse(filepath.FilePath(b'z') != filepath.FilePath(b'z'))

    def test_descendantOnly(self) -> None:
        """
        If C{".."} is in the sequence passed to L{FilePath.descendant},
        L{InsecurePath} is raised.
        """
        self.assertRaises(filepath.InsecurePath, self.path.descendant, ['mon€y', '..'])

    def testSibling(self) -> None:
        p = self.path.child(b'sibling_start')
        ts = p.sibling(b'sibling_test')
        self.assertEqual(ts.dirname(), p.dirname())
        self.assertEqual(ts.basename(), b'sibling_test')
        ts.createDirectory()
        self.assertIn(ts, self.path.children())

    def testTemporarySibling(self) -> None:
        ts = self.path.temporarySibling()
        self.assertEqual(ts.dirname(), self.path.dirname())
        self.assertNotIn(ts.basename(), self.path.listdir())
        ts.createDirectory()
        self.assertIn(ts, self.path.parent().children())

    def test_temporarySiblingExtension(self) -> None:
        """
        If L{FilePath.temporarySibling} is given an extension argument, it will
        produce path objects with that extension appended to their names.
        """
        testExtension = b'.test-extension'
        ts = self.path.temporarySibling(testExtension)
        self.assertTrue(ts.basename().endswith(testExtension), f'{ts.basename()!r} does not end with {testExtension!r}')

    def test_removeDirectory(self) -> None:
        """
        L{FilePath.remove} on a L{FilePath} that refers to a directory will
        recursively delete its contents.
        """
        self.path.remove()
        self.assertFalse(self.path.exists())

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_removeWithSymlink(self) -> None:
        """
        For a path which is a symbolic link, L{FilePath.remove} just deletes
        the link, not the target.
        """
        link = self.path.child(b'sub1.link')
        os.symlink(self.path.child(b'sub1').path, link.path)
        link.remove()
        self.assertFalse(link.exists())
        self.assertTrue(self.path.child(b'sub1').exists())

    def test_copyToDirectory(self) -> None:
        """
        L{FilePath.copyTo} makes a copy of all the contents of the directory
        named by that L{FilePath} if it is able to do so.
        """
        oldPaths = list(self.path.walk())
        fp = filepath.FilePath(self.mktemp())
        self.path.copyTo(fp)
        self.path.remove()
        fp.copyTo(self.path)
        newPaths = list(self.path.walk())
        newPaths.sort()
        oldPaths.sort()
        self.assertEqual(newPaths, oldPaths)

    def test_copyToMissingDestFileClosing(self) -> None:
        """
        If an exception is raised while L{FilePath.copyTo} is trying to open
        source file to read from, the destination file is closed and the
        exception is raised to the caller of L{FilePath.copyTo}.
        """
        nosuch = self.path.child(b'nothere')
        nosuch.isfile = lambda: True
        destination = ExplodingFilePath(self.mktemp())
        self.assertRaises(IOError, nosuch.copyTo, destination)
        self.assertTrue(destination.fp.closed)

    def test_copyToFileClosing(self) -> None:
        """
        If an exception is raised while L{FilePath.copyTo} is copying bytes
        between two regular files, the source and destination files are closed
        and the exception propagates to the caller of L{FilePath.copyTo}.
        """
        destination = ExplodingFilePath(self.mktemp())
        source = ExplodingFilePath(__file__)
        self.assertRaises(IOError, source.copyTo, destination)
        self.assertTrue(source.fp.closed)
        self.assertTrue(destination.fp.closed)

    def test_copyToDirectoryItself(self) -> None:
        """
        L{FilePath.copyTo} fails with an OSError or IOError (depending on
        platform, as it propagates errors from open() and write()) when
        attempting to copy a directory to a child of itself.
        """
        self.assertRaises((OSError, IOError), self.path.copyTo, self.path.child(b'file1'))

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_copyToWithSymlink(self) -> None:
        """
        Verify that copying with followLinks=True copies symlink targets
        instead of symlinks
        """
        os.symlink(self.path.child(b'sub1').path, self.path.child(b'link1').path)
        fp = filepath.FilePath(self.mktemp())
        self.path.copyTo(fp)
        self.assertFalse(fp.child(b'link1').islink())
        self.assertEqual([x.basename() for x in fp.child(b'sub1').children()], [x.basename() for x in fp.child(b'link1').children()])

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_copyToWithoutSymlink(self) -> None:
        """
        Verify that copying with followLinks=False copies symlinks as symlinks
        """
        os.symlink(b'sub1', self.path.child(b'link1').path)
        fp = filepath.FilePath(self.mktemp())
        self.path.copyTo(fp, followLinks=False)
        self.assertTrue(fp.child(b'link1').islink())
        self.assertEqual(os.readlink(self.path.child(b'link1').path), os.readlink(fp.child(b'link1').path))

    def test_copyToMissingSource(self) -> None:
        """
        If the source path is missing, L{FilePath.copyTo} raises L{OSError}.
        """
        path = filepath.FilePath(self.mktemp())
        exc = self.assertRaises(OSError, path.copyTo, b'some other path')
        self.assertEqual(exc.errno, errno.ENOENT)

    def test_moveTo(self) -> None:
        """
        Verify that moving an entire directory results into another directory
        with the same content.
        """
        oldPaths = list(self.path.walk())
        fp = filepath.FilePath(self.mktemp())
        self.path.moveTo(fp)
        fp.moveTo(self.path)
        newPaths = list(self.path.walk())
        newPaths.sort()
        oldPaths.sort()
        self.assertEqual(newPaths, oldPaths)

    def test_moveToExistsCache(self) -> None:
        """
        A L{FilePath} that has been moved aside with L{FilePath.moveTo} no
        longer registers as existing.  Its previously non-existent target
        exists, though, as it was created by the call to C{moveTo}.
        """
        fp = filepath.FilePath(self.mktemp())
        fp2 = filepath.FilePath(self.mktemp())
        fp.touch()
        self.assertTrue(fp.exists())
        self.assertFalse(fp2.exists())
        fp.moveTo(fp2)
        self.assertFalse(fp.exists())
        self.assertTrue(fp2.exists())

    def test_moveToExistsCacheCrossMount(self) -> None:
        """
        The assertion of test_moveToExistsCache should hold in the case of a
        cross-mount move.
        """
        self.setUpFaultyRename()
        self.test_moveToExistsCache()

    def test_moveToSizeCache(self, hook: Callable[[], None]=lambda: None) -> None:
        """
        L{FilePath.moveTo} clears its destination's status cache, such that
        calls to L{FilePath.getsize} after the call to C{moveTo} will report the
        new size, not the old one.

        This is a separate test from C{test_moveToExistsCache} because it is
        intended to cover the fact that the destination's cache is dropped;
        test_moveToExistsCache doesn't cover this case because (currently) a
        file that doesn't exist yet does not cache the fact of its non-
        existence.
        """
        fp = filepath.FilePath(self.mktemp())
        fp2 = filepath.FilePath(self.mktemp())
        fp.setContent(b'1234')
        fp2.setContent(b'1234567890')
        hook()
        self.assertEqual(fp.getsize(), 4)
        self.assertEqual(fp2.getsize(), 10)
        os.remove(fp2.path)
        self.assertEqual(fp2.getsize(), 10)
        fp.moveTo(fp2)
        self.assertEqual(fp2.getsize(), 4)

    def test_moveToSizeCacheCrossMount(self) -> None:
        """
        The assertion of test_moveToSizeCache should hold in the case of a
        cross-mount move.
        """

        def setUp() -> None:
            self.setUpFaultyRename()
        self.test_moveToSizeCache(hook=setUp)

    def test_moveToError(self) -> None:
        """
        Verify error behavior of moveTo: it should raises one of OSError or
        IOError if you want to move a path into one of its child. It's simply
        the error raised by the underlying rename system call.
        """
        self.assertRaises((OSError, IOError), self.path.moveTo, self.path.child(b'file1'))

    def setUpFaultyRename(self) -> List[Tuple[str, str]]:
        """
        Set up a C{os.rename} that will fail with L{errno.EXDEV} on first call.
        This is used to simulate a cross-device rename failure.

        @return: a list of pair (src, dest) of calls to C{os.rename}
        @rtype: C{list} of C{tuple}
        """
        invokedWith = []

        def faultyRename(src: str, dest: str) -> None:
            invokedWith.append((src, dest))
            if len(invokedWith) == 1:
                raise OSError(errno.EXDEV, 'Test-induced failure simulating cross-device rename failure')
            return originalRename(src, dest)
        originalRename = os.rename
        self.patch(os, 'rename', faultyRename)
        return invokedWith

    def test_crossMountMoveTo(self) -> None:
        """
        C{moveTo} should be able to handle C{EXDEV} error raised by
        C{os.rename} when trying to move a file on a different mounted
        filesystem.
        """
        invokedWith = self.setUpFaultyRename()
        self.test_moveTo()
        self.assertTrue(invokedWith)

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_crossMountMoveToWithSymlink(self) -> None:
        """
        By default, when moving a symlink, it should follow the link and
        actually copy the content of the linked node.
        """
        invokedWith = self.setUpFaultyRename()
        f2 = self.path.child(b'file2')
        f3 = self.path.child(b'file3')
        os.symlink(self.path.child(b'file1').path, f2.path)
        f2.moveTo(f3)
        self.assertFalse(f3.islink())
        self.assertEqual(f3.getContent(), b'file 1')
        self.assertTrue(invokedWith)

    @skipIf(symlinkSkip, 'Platform does not support symlinks')
    def test_crossMountMoveToWithoutSymlink(self) -> None:
        """
        Verify that moveTo called with followLinks=False actually create
        another symlink.
        """
        invokedWith = self.setUpFaultyRename()
        f2 = self.path.child(b'file2')
        f3 = self.path.child(b'file3')
        os.symlink(self.path.child(b'file1').path, f2.path)
        f2.moveTo(f3, followLinks=False)
        self.assertTrue(f3.islink())
        self.assertEqual(f3.getContent(), b'file 1')
        self.assertTrue(invokedWith)

    def test_createBinaryMode(self) -> None:
        """
        L{FilePath.create} should always open (and write to) files in binary
        mode; line-feed octets should be unmodified.

        (While this test should pass on all platforms, it is only really
        interesting on platforms which have the concept of binary mode, i.e.
        Windows platforms.)
        """
        path = filepath.FilePath(self.mktemp())
        with path.create() as f:
            self.assertIn('b', f.mode)
            f.write(b'\n')
        with open(path.path, 'rb') as fp:
            read = fp.read()
            self.assertEqual(read, b'\n')

    def testOpen(self) -> None:
        nonexistent = self.path.child(b'nonexistent')
        e = self.assertRaises(IOError, nonexistent.open)
        self.assertEqual(e.errno, errno.ENOENT)
        writer = self.path.child(b'writer')
        with writer.open('w') as f:
            f.write(b'abc\ndef')
        with writer.open() as f:
            self.assertEqual(f.read(), b'abc\ndef')
        writer.open('w').close()
        with writer.open() as f:
            self.assertEqual(f.read(), b'')
        appender = self.path.child(b'appender')
        with appender.open('w') as f:
            f.write(b'abc')
        with appender.open('a') as f:
            f.write(b'def')
        with appender.open('r') as f:
            self.assertEqual(f.read(), b'abcdef')
        with appender.open('r+') as f:
            self.assertEqual(f.read(), b'abcdef')
            f.seek(0, 1)
            f.write(b'ghi')
        with appender.open('r') as f:
            self.assertEqual(f.read(), b'abcdefghi')
        with appender.open('w+') as f:
            self.assertEqual(f.read(), b'')
            f.seek(0, 1)
            f.write(b'123')
        with appender.open('a+') as f:
            f.write(b'456')
            f.seek(0, 1)
            self.assertEqual(f.read(), b'')
            f.seek(0, 0)
            self.assertEqual(f.read(), b'123456')
        nonexistent.requireCreate(True)
        nonexistent.open('w').close()
        existent = nonexistent
        del nonexistent
        self.assertRaises((OSError, IOError), existent.open)

    def test_openWithExplicitBinaryMode(self) -> None:
        """
        Due to a bug in Python 2.7 on Windows including multiple 'b'
        characters in the mode passed to the built-in open() will cause an
        error.  FilePath.open() ensures that only a single 'b' character is
        included in the mode passed to the built-in open().

        See http://bugs.python.org/issue7686 for details about the bug.
        """
        writer = self.path.child(b'explicit-binary')
        opener = writer.open('wb')
        with opener as file:
            file.write(b'abc\ndef')
        self.assertTrue(writer.exists)

    def test_openWithRedundantExplicitBinaryModes(self) -> None:
        """
        Due to a bug in Python 2.7 on Windows including multiple 'b'
        characters in the mode passed to the built-in open() will cause an
        error.  No matter how many 'b' modes are specified, FilePath.open()
        ensures that only a single 'b' character is included in the mode
        passed to the built-in open().

        See http://bugs.python.org/issue7686 for details about the bug.
        """
        writer = self.path.child(b'multiple-binary')
        opener = writer.open('wbb')
        with opener as file:
            file.write(b'abc\ndef')
        self.assertTrue(writer.exists)

    def test_existsCache(self) -> None:
        """
        Check that C{filepath.FilePath.exists} correctly restat the object if
        an operation has occurred in the mean time.
        """
        fp = filepath.FilePath(self.mktemp())
        self.assertFalse(fp.exists())
        fp.makedirs()
        self.assertTrue(fp.exists())

    def test_makedirsMakesDirectoriesRecursively(self) -> None:
        """
        C{FilePath.makedirs} creates a directory at C{path}}, including
        recursively creating all parent directories leading up to the path.
        """
        fp = filepath.FilePath(os.path.join(self.mktemp(), b'foo', b'bar', b'baz'))
        self.assertFalse(fp.exists())
        fp.makedirs()
        self.assertTrue(fp.exists())
        self.assertTrue(fp.isdir())

    def test_makedirsMakesDirectoriesWithIgnoreExistingDirectory(self) -> None:
        """
        Calling C{FilePath.makedirs} with C{ignoreExistingDirectory} set to
        C{True} has no effect if directory does not exist.
        """
        fp = filepath.FilePath(self.mktemp())
        self.assertFalse(fp.exists())
        fp.makedirs(ignoreExistingDirectory=True)
        self.assertTrue(fp.exists())
        self.assertTrue(fp.isdir())

    def test_makedirsThrowsWithExistentDirectory(self) -> None:
        """
        C{FilePath.makedirs} throws an C{OSError} exception
        when called on a directory that already exists.
        """
        fp = filepath.FilePath(os.path.join(self.mktemp()))
        fp.makedirs()
        exception = self.assertRaises(OSError, fp.makedirs)
        self.assertEqual(exception.errno, errno.EEXIST)

    def test_makedirsAcceptsIgnoreExistingDirectory(self) -> None:
        """
        C{FilePath.makedirs} succeeds when called on a directory that already
        exists and the c{ignoreExistingDirectory} argument is set to C{True}.
        """
        fp = filepath.FilePath(self.mktemp())
        fp.makedirs()
        self.assertTrue(fp.exists())
        fp.makedirs(ignoreExistingDirectory=True)
        self.assertTrue(fp.exists())

    def test_makedirsIgnoreExistingDirectoryExistAlreadyAFile(self) -> None:
        """
        When C{FilePath.makedirs} is called with C{ignoreExistingDirectory} set
        to C{True} it throws an C{OSError} exceptions if path is a file.
        """
        fp = filepath.FilePath(self.mktemp())
        fp.create()
        self.assertTrue(fp.isfile())
        exception = self.assertRaises(OSError, fp.makedirs, ignoreExistingDirectory=True)
        self.assertEqual(exception.errno, errno.EEXIST)

    def test_makedirsRaisesNonEexistErrorsIgnoreExistingDirectory(self) -> None:
        """
        When C{FilePath.makedirs} is called with C{ignoreExistingDirectory} set
        to C{True} it raises an C{OSError} exception if exception errno is not
        EEXIST.
        """

        def faultyMakedirs(path: str) -> None:
            raise OSError(errno.EACCES, 'Permission Denied')
        self.patch(os, 'makedirs', faultyMakedirs)
        fp = filepath.FilePath(self.mktemp())
        exception = self.assertRaises(OSError, fp.makedirs, ignoreExistingDirectory=True)
        self.assertEqual(exception.errno, errno.EACCES)

    def test_changed(self) -> None:
        """
        L{FilePath.changed} indicates that the L{FilePath} has changed, but does
        not re-read the status information from the filesystem until it is
        queried again via another method, such as C{getsize}.
        """
        fp = filepath.FilePath(self.mktemp())
        fp.setContent(b'12345')
        self.assertEqual(fp.getsize(), 5)
        with open(fp.path, 'wb') as fObj:
            fObj.write(b'12345678')
        self.assertEqual(fp.getsize(), 5)
        fp.changed()
        self.assertEqual(fp.getsize(), 8)

    @skipIf(platform.isWindows(), 'Test does not run on Windows')
    def test_getPermissions_POSIX(self) -> None:
        """
        Getting permissions for a file returns a L{Permissions} object for
        POSIX platforms (which supports separate user, group, and other
        permissions bits.
        """
        for mode in (511, 448):
            self.path.child(b'sub1').chmod(mode)
            self.assertEqual(self.path.child(b'sub1').getPermissions(), filepath.Permissions(mode))
        self.path.child(b'sub1').chmod(500)
        self.assertEqual(self.path.child(b'sub1').getPermissions().shorthand(), 'rwxrw-r--')

    def test_filePathNotDeprecated(self) -> None:
        """
        While accessing L{twisted.python.filepath.FilePath.statinfo} is
        deprecated, the filepath itself is not.
        """
        filepath.FilePath(self.mktemp())
        warningInfo = self.flushWarnings([self.test_filePathNotDeprecated])
        self.assertEqual(warningInfo, [])

    @skipIf(not platform.isWindows(), 'Test will run only on Windows')
    def test_getPermissions_Windows(self) -> None:
        """
        Getting permissions for a file returns a L{Permissions} object in
        Windows.  Windows requires a different test, because user permissions
        = group permissions = other permissions.  Also, chmod may not be able
        to set the execute bit, so we are skipping tests that set the execute
        bit.
        """
        self.addCleanup(self.path.child(b'sub1').chmod, 511)
        for mode in (511, 365):
            self.path.child(b'sub1').chmod(mode)
            self.assertEqual(self.path.child(b'sub1').getPermissions(), filepath.Permissions(mode))
        self.path.child(b'sub1').chmod(329)
        self.assertEqual(self.path.child(b'sub1').getPermissions().shorthand(), 'r-xr-xr-x')

    def test_whetherBlockOrSocket(self) -> None:
        """
        Ensure that a file is not a block or socket
        """
        self.assertFalse(self.path.isBlockDevice())
        self.assertFalse(self.path.isSocket())

    @skipIf(not platform.isWindows(), 'Test will run only on Windows')
    def test_statinfoBitsNotImplementedInWindows(self) -> None:
        """
        Verify that certain file stats are not available on Windows
        """
        self.assertRaises(NotImplementedError, self.path.getInodeNumber)
        self.assertRaises(NotImplementedError, self.path.getDevice)
        self.assertRaises(NotImplementedError, self.path.getNumberOfHardLinks)
        self.assertRaises(NotImplementedError, self.path.getUserID)
        self.assertRaises(NotImplementedError, self.path.getGroupID)

    @skipIf(platform.isWindows(), 'Test does not run on Windows')
    def test_statinfoBitsAreNumbers(self) -> None:
        """
        Verify that file inode/device/nlinks/uid/gid stats are numbers in
        a POSIX environment
        """
        c = self.path.child(b'file1')
        for p in (self.path, c):
            self.assertIsInstance(p.getInodeNumber(), int)
            self.assertIsInstance(p.getDevice(), int)
            self.assertIsInstance(p.getNumberOfHardLinks(), int)
            self.assertIsInstance(p.getUserID(), int)
            self.assertIsInstance(p.getGroupID(), int)
        self.assertEqual(self.path.getUserID(), c.getUserID())
        self.assertEqual(self.path.getGroupID(), c.getGroupID())

    @skipIf(platform.isWindows(), 'Test does not run on Windows')
    def test_statinfoNumbersAreValid(self) -> None:
        """
        Verify that the right numbers come back from the right accessor methods
        for file inode/device/nlinks/uid/gid (in a POSIX environment)
        """

        class FakeStat:
            st_ino = 200
            st_dev = 300
            st_nlink = 400
            st_uid = 500
            st_gid = 600
        fake = FakeStat()

        def fakeRestat(*args: object, **kwargs: object) -> None:
            self.path._statinfo = fake
        self.path.restat = fakeRestat
        self.path._statinfo = None
        self.assertEqual(self.path.getInodeNumber(), fake.st_ino)
        self.assertEqual(self.path.getDevice(), fake.st_dev)
        self.assertEqual(self.path.getNumberOfHardLinks(), fake.st_nlink)
        self.assertEqual(self.path.getUserID(), fake.st_uid)
        self.assertEqual(self.path.getGroupID(), fake.st_gid)