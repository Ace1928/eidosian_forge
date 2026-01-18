import os
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict
from zope.interface import Interface, implementer
from twisted.python.compat import execfile
class NoDocumentsFound(Exception):
    """
    Raised when no input documents are found.
    """