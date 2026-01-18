import os
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict
from zope.interface import Interface, implementer
from twisted.python.compat import execfile
@implementer(IVCSCommand)
class GitCommand:
    """
    Subset of Git commands to release Twisted from a Git repository.
    """

    @staticmethod
    def ensureIsWorkingDirectory(path):
        """
        Ensure that C{path} is a Git working directory.

        @type path: L{twisted.python.filepath.FilePath}
        @param path: The path to check.
        """
        try:
            runCommand(['git', 'rev-parse'], cwd=path.path)
        except (CalledProcessError, OSError):
            raise NotWorkingDirectory(f'{path.path} does not appear to be a Git repository.')

    @staticmethod
    def isStatusClean(path):
        """
        Return the Git status of the files in the specified path.

        @type path: L{twisted.python.filepath.FilePath}
        @param path: The path to get the status from (can be a directory or a
            file.)
        """
        status = runCommand(['git', '-C', path.path, 'status', '--short']).strip()
        return status == b''

    @staticmethod
    def remove(path):
        """
        Remove the specified path from a Git repository.

        @type path: L{twisted.python.filepath.FilePath}
        @param path: The path to remove from the repository.
        """
        runCommand(['git', '-C', path.dirname(), 'rm', path.path])

    @staticmethod
    def exportTo(fromDir, exportDir):
        """
        Export the content of a Git repository to the specified directory.

        @type fromDir: L{twisted.python.filepath.FilePath}
        @param fromDir: The path to the Git repository to export.

        @type exportDir: L{twisted.python.filepath.FilePath}
        @param exportDir: The directory to export the content of the
            repository to. This directory doesn't have to exist prior to
            exporting the repository.
        """
        runCommand(['git', '-C', fromDir.path, 'checkout-index', '--all', '--force', '--prefix', exportDir.path + '/'])