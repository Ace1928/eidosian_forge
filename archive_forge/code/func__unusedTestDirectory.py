from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def _unusedTestDirectory(base):
    """
    Find an unused directory named similarly to C{base}.

    Once a directory is found, it will be locked and a marker dropped into it
    to identify it as a trial temporary directory.

    @param base: A template path for the discovery process.  If this path
        exactly cannot be used, a path which varies only in a suffix of the
        basename will be used instead.
    @type base: L{FilePath}

    @return: A two-tuple.  The first element is a L{FilePath} representing the
        directory which was found and created.  The second element is a locked
        L{FilesystemLock<twisted.python.lockfile.FilesystemLock>}.  Another
        call to C{_unusedTestDirectory} will not be able to reused the
        same name until the lock is released, either explicitly or by this
        process exiting.
    """
    counter = 0
    while True:
        if counter:
            testdir = base.sibling('%s-%d' % (base.basename(), counter))
        else:
            testdir = base
        testdir.parent().makedirs(ignoreExistingDirectory=True)
        testDirLock = FilesystemLock(testdir.path + '.lock')
        if testDirLock.lock():
            if testdir.exists():
                _removeSafely(testdir)
            testdir.makedirs()
            testdir.child(b'_trial_marker').setContent(b'')
            return (testdir, testDirLock)
        elif base.basename() == '_trial_temp':
            counter += 1
        else:
            raise _WorkingDirectoryBusy()