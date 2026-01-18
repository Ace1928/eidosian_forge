from __future__ import annotations
import inspect
import sys
import warnings
import zipimport
from os.path import dirname, split as splitpath
from zope.interface import Interface, implementer
from twisted.python.compat import nativeString
from twisted.python.components import registerAdapter
from twisted.python.filepath import FilePath, UnlistableError
from twisted.python.reflect import namedAny
from twisted.python.zippath import ZipArchive
def _isPythonIdentifier(string):
    """
    cheezy fake test for proper identifier-ness.

    @param string: a L{str} which might or might not be a valid python
        identifier.
    @return: True or False
    """
    textString = nativeString(string)
    return ' ' not in textString and '.' not in textString and ('-' not in textString)