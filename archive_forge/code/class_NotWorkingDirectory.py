import os
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict
from zope.interface import Interface, implementer
from twisted.python.compat import execfile
class NotWorkingDirectory(Exception):
    """
    Raised when a directory does not appear to be a repository directory of a
    supported VCS.
    """