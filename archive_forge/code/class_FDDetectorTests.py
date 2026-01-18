import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
class FDDetectorTests(TestCase):
    """
    Tests for _FDDetector class in twisted.internet.process, which detects
    which function to drop in place for the _listOpenFDs method.

    @ivar devfs: A flag indicating whether the filesystem fake will indicate
        that /dev/fd exists.

    @ivar accurateDevFDResults: A flag indicating whether the /dev/fd fake
        returns accurate open file information.

    @ivar procfs: A flag indicating whether the filesystem fake will indicate
        that /proc/<pid>/fd exists.
    """
    skip = platformSkip
    devfs = False
    accurateDevFDResults = False
    procfs = False

    def getpid(self):
        """
        Fake os.getpid, always return the same thing
        """
        return 123

    def listdir(self, arg):
        """
        Fake os.listdir, depending on what mode we're in to simulate behaviour.

        @param arg: the directory to list
        """
        accurate = map(str, self._files)
        if self.procfs and arg == '/proc/%d/fd' % (self.getpid(),):
            return accurate
        if self.devfs and arg == '/dev/fd':
            if self.accurateDevFDResults:
                return accurate
            return ['0', '1', '2']
        raise OSError()

    def openfile(self, fname, mode):
        """
        This is a mock for L{open}.  It keeps track of opened files so extra
        descriptors can be returned from the mock for L{os.listdir} when used on
        one of the list-of-filedescriptors directories.

        A L{FakeFile} is returned which can be closed to remove the new
        descriptor from the open list.
        """
        f = FakeFile(self, min(set(range(1024)) - set(self._files)))
        self._files.append(f.fd)
        return f

    def hideResourceModule(self):
        """
        Make the L{resource} module unimportable for the remainder of the
        current test method.
        """
        sys.modules['resource'] = None

    def revealResourceModule(self, limit):
        """
        Make a L{FakeResourceModule} instance importable at the L{resource}
        name.

        @param limit: The value which will be returned for the hard limit of
            number of open files by the fake resource module's C{getrlimit}
            function.
        """
        sys.modules['resource'] = FakeResourceModule(limit)

    def replaceResourceModule(self, value):
        """
        Restore the original resource module to L{sys.modules}.
        """
        if value is None:
            try:
                del sys.modules['resource']
            except KeyError:
                pass
        else:
            sys.modules['resource'] = value

    def setUp(self):
        """
        Set up the tests, giving ourselves a detector object to play with and
        setting up its testable knobs to refer to our mocked versions.
        """
        self.detector = process._FDDetector()
        self.detector.listdir = self.listdir
        self.detector.getpid = self.getpid
        self.detector.openfile = self.openfile
        self._files = [0, 1, 2]
        self.addCleanup(self.replaceResourceModule, sys.modules.get('resource'))

    def test_selectFirstWorking(self):
        """
        L{FDDetector._getImplementation} returns the first method from its
        C{_implementations} list which returns results which reflect a newly
        opened file descriptor.
        """

        def failWithException():
            raise ValueError('This does not work')

        def failWithWrongResults():
            return [0, 1, 2]

        def correct():
            return self._files[:]
        self.detector._implementations = [failWithException, failWithWrongResults, correct]
        self.assertIs(correct, self.detector._getImplementation())

    def test_selectLast(self):
        """
        L{FDDetector._getImplementation} returns the last method from its
        C{_implementations} list if none of the implementations manage to return
        results which reflect a newly opened file descriptor.
        """

        def failWithWrongResults():
            return [3, 5, 9]

        def failWithOtherWrongResults():
            return [0, 1, 2]
        self.detector._implementations = [failWithWrongResults, failWithOtherWrongResults]
        self.assertIs(failWithOtherWrongResults, self.detector._getImplementation())

    def test_identityOfListOpenFDsChanges(self):
        """
        Check that the identity of _listOpenFDs changes after running
        _listOpenFDs the first time, but not after the second time it's run.

        In other words, check that the monkey patching actually works.
        """
        detector = process._FDDetector()
        first = detector._listOpenFDs.__name__
        detector._listOpenFDs()
        second = detector._listOpenFDs.__name__
        detector._listOpenFDs()
        third = detector._listOpenFDs.__name__
        self.assertNotEqual(first, second)
        self.assertEqual(second, third)

    def test_devFDImplementation(self):
        """
        L{_FDDetector._devFDImplementation} raises L{OSError} if there is no
        I{/dev/fd} directory, otherwise it returns the basenames of its children
        interpreted as integers.
        """
        self.devfs = False
        self.assertRaises(OSError, self.detector._devFDImplementation)
        self.devfs = True
        self.accurateDevFDResults = False
        self.assertEqual([0, 1, 2], self.detector._devFDImplementation())

    def test_procFDImplementation(self):
        """
        L{_FDDetector._procFDImplementation} raises L{OSError} if there is no
        I{/proc/<pid>/fd} directory, otherwise it returns the basenames of its
        children interpreted as integers.
        """
        self.procfs = False
        self.assertRaises(OSError, self.detector._procFDImplementation)
        self.procfs = True
        self.assertEqual([0, 1, 2], self.detector._procFDImplementation())

    def test_resourceFDImplementation(self):
        """
        L{_FDDetector._fallbackFDImplementation} uses the L{resource} module if
        it is available, returning a range of integers from 0 to the
        minimum of C{1024} and the hard I{NOFILE} limit.
        """
        self.revealResourceModule(512)
        self.assertEqual(list(range(512)), list(self.detector._fallbackFDImplementation()))
        self.revealResourceModule(2048)
        self.assertEqual(list(range(1024)), list(self.detector._fallbackFDImplementation()))

    def test_fallbackFDImplementation(self):
        """
        L{_FDDetector._fallbackFDImplementation}, the implementation of last
        resort, succeeds with a fixed range of integers from 0 to 1024 when the
        L{resource} module is not importable.
        """
        self.hideResourceModule()
        self.assertEqual(list(range(1024)), list(self.detector._fallbackFDImplementation()))