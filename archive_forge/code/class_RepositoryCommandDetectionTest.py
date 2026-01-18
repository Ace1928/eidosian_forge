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
class RepositoryCommandDetectionTest(ExternalTempdirTestCase):
    """
    Test the L{getRepositoryCommand} to access the right set of VCS commands
    depending on the repository manipulated.
    """

    def setUp(self):
        self.repos = FilePath(self.mktemp())

    def test_git(self):
        """
        L{getRepositoryCommand} from a Git repository returns L{GitCommand}.
        """
        _gitInit(self.repos)
        cmd = getRepositoryCommand(self.repos)
        self.assertIs(cmd, GitCommand)

    def test_unknownRepository(self):
        """
        L{getRepositoryCommand} from a directory which doesn't look like a Git
        repository produces a L{NotWorkingDirectory} exception.
        """
        self.assertRaises(NotWorkingDirectory, getRepositoryCommand, self.repos)