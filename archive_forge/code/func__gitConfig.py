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
def _gitConfig(path):
    """
    Set some config in the repo that Git requires to make commits. This isn't
    needed in real usage, just for tests.

    @param path: The path to the Git repository.
    @type path: L{FilePath}
    """
    runCommand(['git', 'config', '--file', path.child('.git').child('config').path, 'user.name', '"someone"'])
    runCommand(['git', 'config', '--file', path.child('.git').child('config').path, 'user.email', '"someone@someplace.com"'])