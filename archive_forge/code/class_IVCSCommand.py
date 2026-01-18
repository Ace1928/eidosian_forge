import os
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict
from zope.interface import Interface, implementer
from twisted.python.compat import execfile
class IVCSCommand(Interface):
    """
    An interface for VCS commands.
    """

    def ensureIsWorkingDirectory(path):
        """
        Ensure that C{path} is a working directory of this VCS.

        @type path: L{twisted.python.filepath.FilePath}
        @param path: The path to check.
        """

    def isStatusClean(path):
        """
        Return the Git status of the files in the specified path.

        @type path: L{twisted.python.filepath.FilePath}
        @param path: The path to get the status from (can be a directory or a
            file.)
        """

    def remove(path):
        """
        Remove the specified path from a the VCS.

        @type path: L{twisted.python.filepath.FilePath}
        @param path: The path to remove from the repository.
        """

    def exportTo(fromDir, exportDir):
        """
        Export the content of the VCSrepository to the specified directory.

        @type fromDir: L{twisted.python.filepath.FilePath}
        @param fromDir: The path to the VCS repository to export.

        @type exportDir: L{twisted.python.filepath.FilePath}
        @param exportDir: The directory to export the content of the
            repository to. This directory doesn't have to exist prior to
            exporting the repository.
        """